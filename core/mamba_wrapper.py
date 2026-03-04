"""
Mamba Model Wrapper for Spectral Analysis.

This module provides a unified interface for loading and interacting with
Mamba SSM models, supporting both the official mamba-ssm library and
the mamba-minimal implementation.

References:
    [Gu2023] Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence Modeling
             with Selective State Spaces. arXiv:2312.00752
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class MambaWrapper:
    """
    Unified wrapper for Mamba SSM models.
    
    Provides a consistent interface for loading pretrained models and
    accessing internal Mamba blocks for spectral analysis.
    
    Attributes:
        model: The underlying Mamba model.
        device: Device where the model is loaded ('cuda' or 'cpu').
        model_name: Name of the loaded model.
        
    Example:
        >>> wrapper = MambaWrapper.load_pretrained("state-spaces/mamba-130m")
        >>> layers = wrapper.get_mamba_layers()
        >>> print(f"Found {len(layers)} Mamba layers")
        
    Note:
        This wrapper supports both mamba-ssm official models and
        mamba-minimal implementations with automatic detection.
    """
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: Optional[Any] = None,
        device: str = "cuda",
        model_name: str = "unknown",
    ) -> None:
        """
        Initialize wrapper with an existing model.
        
        Args:
            model: Pretrained Mamba model.
            tokenizer: Optional tokenizer for text processing.
            device: Device for computation ('cuda' or 'cpu').
            model_name: Identifier for the model.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model_name = model_name
        self._mamba_layers: Optional[List[nn.Module]] = None
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        
        # Move model to device
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"MambaWrapper initialized with model: {model_name} on {device}")
    
    @classmethod
    def load_pretrained(
        cls,
        model_name: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "MambaWrapper":
        """
        Load a pretrained Mamba model from Hugging Face.
        
        Supports loading models from:
        - Official mamba-ssm library (state-spaces/mamba-*)
        - mamba-minimal implementation
        
        Args:
            model_name: Hugging Face model identifier or local path.
                Examples: "state-spaces/mamba-130m", "state-spaces/mamba-2.8b"
            device: Target device ('cuda' or 'cpu').
            dtype: Data type for model weights.
            
        Returns:
            MambaWrapper: Initialized wrapper with loaded model.
            
        Raises:
            ImportError: If neither mamba-ssm nor mamba-minimal is available.
            ValueError: If model cannot be loaded.
            
        Example:
            >>> wrapper = MambaWrapper.load_pretrained("state-spaces/mamba-130m")
        """
        tokenizer = None
        
        # Try loading with mamba-ssm first
        try:
            from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
            
            logger.info(f"Loading {model_name} with mamba-ssm...")
            model = MambaLMHeadModel.from_pretrained(
                model_name,
                device=device,
                dtype=dtype,
            )
            
            # Load tokenizer from transformers
            try:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            except Exception as e:
                logger.warning(f"Could not load tokenizer: {e}")
            
            return cls(model, tokenizer, device, model_name)
            
        except ImportError:
            logger.info("mamba-ssm not available, trying mamba-minimal...")
        except Exception as e:
            logger.warning(f"Failed to load with mamba-ssm: {e}")
        
        # Try mamba-minimal as fallback
        try:
            # Import from local mamba-minimal if available
            from model import Mamba as MambaMinimal
            from transformers import AutoTokenizer
            
            logger.info(f"Loading {model_name} with mamba-minimal...")
            model = MambaMinimal.from_pretrained(model_name)
            model = model.to(device).to(dtype)
            
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
            
            return cls(model, tokenizer, device, model_name)
            
        except ImportError:
            pass
        except Exception as e:
            logger.warning(f"Failed to load with mamba-minimal: {e}")
        
        raise ImportError(
            "Neither mamba-ssm nor mamba-minimal is available. "
            "Please install one of them:\n"
            "  pip install mamba-ssm  # Official (requires CUDA)\n"
            "  # OR clone https://github.com/johnma2006/mamba-minimal"
        )
    
    @classmethod
    def from_config(
        cls,
        d_model: int = 768,
        n_layers: int = 24,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        device: str = "cuda",
        **kwargs,
    ) -> "MambaWrapper":
        """
        Create a new (untrained) Mamba model from configuration.
        
        Useful for testing and experimentation without loading
        pretrained weights.
        
        Args:
            d_model: Model dimension.
            n_layers: Number of Mamba layers.
            d_state: SSM state expansion factor.
            d_conv: Local convolution width.
            expand: Block expansion factor.
            device: Target device.
            **kwargs: Additional configuration passed to Mamba.
            
        Returns:
            MambaWrapper: Wrapper with randomly initialized model.
        """
        try:
            from mamba_ssm import Mamba
            
            # Create a simple sequential model
            layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    **kwargs,
                )
                for _ in range(n_layers)
            ])
            
            model = nn.Sequential(*layers)
            return cls(model, None, device, f"custom-{d_model}x{n_layers}")
            
        except ImportError:
            raise ImportError(
                "mamba-ssm is required to create models from config. "
                "Install with: pip install mamba-ssm"
            )
    
    def get_mamba_layers(self) -> List[nn.Module]:
        """
        Extract all Mamba blocks from the model.
        
        Returns:
            List[nn.Module]: List of Mamba layer modules.
            
        Note:
            Results are cached after first call.
        """
        if self._mamba_layers is not None:
            return self._mamba_layers
        
        layers = []
        
        # Try different model structures
        # 1. MambaLMHeadModel structure (official)
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layers"):
            for layer in self.model.backbone.layers:
                if hasattr(layer, "mixer"):
                    layers.append(layer.mixer)
                    
        # 2. Direct Sequential structure
        elif isinstance(self.model, nn.Sequential):
            for module in self.model:
                layers.append(module)
                
        # 3. ModuleList structure
        elif hasattr(self.model, "layers"):
            for layer in self.model.layers:
                if hasattr(layer, "mixer"):
                    layers.append(layer.mixer)
                else:
                    layers.append(layer)
        
        # 4. Recursive search fallback
        else:
            layers = self._find_mamba_layers_recursive(self.model)
        
        self._mamba_layers = layers
        logger.info(f"Found {len(layers)} Mamba layers")
        
        return layers
    
    def _find_mamba_layers_recursive(
        self,
        module: nn.Module,
        target_class_names: Tuple[str, ...] = ("Mamba", "Mamba2", "MambaBlock"),
    ) -> List[nn.Module]:
        """
        Recursively find Mamba layers in the model.
        
        Args:
            module: Module to search.
            target_class_names: Class names to look for.
            
        Returns:
            List of found Mamba modules.
        """
        layers = []
        
        for name, child in module.named_children():
            class_name = child.__class__.__name__
            
            if class_name in target_class_names:
                layers.append(child)
            else:
                # Recurse into child modules
                layers.extend(
                    self._find_mamba_layers_recursive(child, target_class_names)
                )
        
        return layers
    
    def register_hook(
        self,
        layer_idx: int,
        hook_fn: Callable[[nn.Module, Tuple, Any], None],
        hook_type: str = "forward",
    ) -> torch.utils.hooks.RemovableHandle:
        """
        Register a hook on a specific Mamba layer.
        
        Args:
            layer_idx: Index of the target layer.
            hook_fn: Hook function to register.
            hook_type: Type of hook ('forward', 'backward', 'forward_pre').
            
        Returns:
            RemovableHandle: Handle to remove the hook later.
        """
        layers = self.get_mamba_layers()
        
        if layer_idx >= len(layers):
            raise IndexError(
                f"Layer index {layer_idx} out of range. "
                f"Model has {len(layers)} layers."
            )
        
        layer = layers[layer_idx]
        
        if hook_type == "forward":
            handle = layer.register_forward_hook(hook_fn)
        elif hook_type == "backward":
            handle = layer.register_full_backward_hook(hook_fn)
        elif hook_type == "forward_pre":
            handle = layer.register_forward_pre_hook(hook_fn)
        else:
            raise ValueError(f"Unknown hook type: {hook_type}")
        
        self._hooks.append(handle)
        return handle
    
    def clear_hooks(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()
        logger.debug("Cleared all hooks")
    
    def tokenize(
        self,
        text: str,
        return_tensors: str = "pt",
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize input text.
        
        Args:
            text: Input text to tokenize.
            return_tensors: Format for returned tensors.
            
        Returns:
            Dictionary with tokenized inputs.
            
        Raises:
            ValueError: If no tokenizer is available.
        """
        if self.tokenizer is None:
            raise ValueError(
                "No tokenizer available. Load model with tokenizer or set manually."
            )
        
        tokens = self.tokenizer(
            text,
            return_tensors=return_tensors,
            padding=False,
            truncation=True,
        )
        
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> str:
        """
        Generate text continuation from prompt.
        
        Args:
            prompt: Input prompt text.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = neutral).
            top_k: If set, only sample from top-k tokens.
            top_p: If set, use nucleus sampling with this probability.
            
        Returns:
            Generated text including the prompt.
            
        Example:
            >>> wrapper.generate("The capital of France is", max_new_tokens=10)
            'The capital of France is Paris, which is also...'
        """
        if self.tokenizer is None:
            raise ValueError("Tokenizer required for text generation")
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
        input_ids = input_ids.to(self.device)
        
        # Check if model has generate method
        if hasattr(self.model, "generate"):
            output_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.shape[1] + max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                cg=True,  # Cache graph for efficiency
            )
        else:
            # Manual generation for simple models
            output_ids = self._generate_manual(
                input_ids, max_new_tokens, temperature, top_k, top_p
            )
        
        return self.tokenizer.decode(output_ids[0])
    
    def _generate_manual(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: Optional[int],
        top_p: Optional[float],
    ) -> torch.Tensor:
        """Manual token-by-token generation."""
        for _ in range(max_new_tokens):
            # Forward pass
            logits = self.model(input_ids)
            
            # Get logits for last token
            next_token_logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    torch.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1]
                sorted_indices_to_remove[..., 0] = False
                indices_to_remove = sorted_indices_to_remove.scatter(
                    1, sorted_indices, sorted_indices_to_remove
                )
                next_token_logits[indices_to_remove] = float("-inf")
            
            # Sample next token
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Input token IDs of shape [batch, seq_len].
            return_hidden_states: If True, return hidden states from each layer.
            
        Returns:
            Model output logits, optionally with hidden states.
        """
        input_ids = input_ids.to(self.device)
        
        if return_hidden_states:
            hidden_states: List[torch.Tensor] = []
            
            def capture_hook(module, input, output):
                if isinstance(output, tuple):
                    hidden_states.append(output[0].detach().cpu())
                else:
                    hidden_states.append(output.detach().cpu())
            
            # Register hooks on all layers
            handles = []
            for layer in self.get_mamba_layers():
                handles.append(layer.register_forward_hook(capture_hook))
            
            try:
                output = self.model(input_ids)
            finally:
                for handle in handles:
                    handle.remove()
            
            return output, hidden_states
        
        return self.model(input_ids)
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Dictionary with model configuration parameters.
        """
        config = {
            "model_name": self.model_name,
            "device": self.device,
            "num_layers": len(self.get_mamba_layers()),
        }
        
        # Try to extract detailed config
        layers = self.get_mamba_layers()
        if layers:
            layer = layers[0]
            if hasattr(layer, "d_model"):
                config["d_model"] = layer.d_model
            if hasattr(layer, "d_state"):
                config["d_state"] = layer.d_state
            if hasattr(layer, "d_conv"):
                config["d_conv"] = layer.d_conv
            if hasattr(layer, "expand"):
                config["expand"] = layer.expand
            if hasattr(layer, "d_inner"):
                config["d_inner"] = layer.d_inner
        
        return config
    
    def __repr__(self) -> str:
        config = self.get_config()
        return (
            f"MambaWrapper(\n"
            f"  model_name='{config['model_name']}',\n"
            f"  device='{config['device']}',\n"
            f"  num_layers={config['num_layers']},\n"
            f")"
        )
