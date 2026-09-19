# Data/sst2.py
"""
SST-2 data loader with support for text corruptions and BERT preprocessing
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
import numpy as np
from datasets import load_dataset
import random
from typing import Optional, Tuple, List

class SST2Dataset(Dataset):
    """SST-2 dataset with support for text corruptions"""
    
    def __init__(self, 
                 split: str = 'train',
                 tokenizer_name: str = 'bert-base-uncased',
                 max_length: int = 128,
                 corruption_type: Optional[str] = None,
                 corruption_severity: float = 0.0):
        """
        Args:
            split: 'train', 'validation', or 'test'
            tokenizer_name: Name of the tokenizer to use
            max_length: Maximum sequence length
            corruption_type: Type of corruption ('swap', 'delete', 'synonym', etc.)
            corruption_severity: Severity of corruption (0.0 to 1.0)
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.corruption_type = corruption_type
        self.corruption_severity = corruption_severity
        
        # Load SST-2 dataset
        dataset = load_dataset('glue', 'sst2')
        self.data = dataset[split]
        
        # Corruption functions
        self.corruption_functions = {
            'swap': self._word_swap_corruption,
            'delete': self._word_delete_corruption,
            'synonym': self._synonym_corruption,
            'keyboard': self._keyboard_typo_corruption
        }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['sentence']
        label = item['label']
        
        # Apply corruption if specified
        if self.corruption_type and self.corruption_severity > 0:
            text = self._apply_corruption(text)
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label)
        }
    
    def _apply_corruption(self, text: str) -> str:
        """Apply specified corruption to text"""
        if self.corruption_type in self.corruption_functions:
            return self.corruption_functions[self.corruption_type](text)
        return text
    
    def _word_swap_corruption(self, text: str) -> str:
        """Randomly swap adjacent words"""
        words = text.split()
        num_swaps = int(len(words) * self.corruption_severity * 0.5)
        
        for _ in range(num_swaps):
            if len(words) > 1:
                idx = random.randint(0, len(words) - 2)
                words[idx], words[idx + 1] = words[idx + 1], words[idx]
        
        return ' '.join(words)
    
    def _word_delete_corruption(self, text: str) -> str:
        """Randomly delete words"""
        words = text.split()
        num_deletions = int(len(words) * self.corruption_severity * 0.3)
        
        for _ in range(num_deletions):
            if len(words) > 1:
                idx = random.randint(0, len(words) - 1)
                words.pop(idx)
        
        return ' '.join(words)
    
    def _synonym_corruption(self, text: str) -> str:
        """Replace words with synonyms (simplified version)"""
        # In practice, use a proper synonym database
        simple_synonyms = {
            'good': ['great', 'excellent', 'fine'],
            'bad': ['terrible', 'awful', 'poor'],
            'movie': ['film', 'picture', 'show']
        }
        
        words = text.split()
        num_replacements = int(len(words) * self.corruption_severity * 0.3)
        
        for _ in range(num_replacements):
            idx = random.randint(0, len(words) - 1)
            word = words[idx].lower()
            if word in simple_synonyms:
                words[idx] = random.choice(simple_synonyms[word])
        
        return ' '.join(words)
    
    def _keyboard_typo_corruption(self, text: str) -> str:
        """Simulate keyboard typos"""
        keyboard_neighbors = {
            'a': ['s', 'q', 'w', 'z'],
            'b': ['v', 'g', 'h', 'n'],
            'c': ['x', 'd', 'f', 'v'],
            'd': ['s', 'x', 'c', 'f'],
            'e': ['w', 'r', 't', 'y'],
            'f': ['d', 'c', 'v', 'g'],
            'g': ['f', 'v', 'b', 'h'],
            'h': ['g', 'b', 'n', 'j'],
            'i': ['u', 'j', 'k', 'l'],
            'j': ['i', 'k', 'l', 'm'],
            'k': ['j', 'l', 'm', 'n'],
            'l': ['k', 'm', 'n', 'o'],
            'm': ['n', 'o', 'p', 'q'],
            'n': ['m', 'o', 'p', 'q'],
            'o': ['n', 'p', 'q', 'r'],
            'p': ['o', 'q', 'r', 's'],
            'q': ['p', 'r', 's', 't'],
            'r': ['e', 't', 'y', 'u'],
            's': ['a', 'd', 'f', 'g'],
            't': ['r', 'y', 'u', 'i'],
            'u': ['y', 'h', 'j', 'k'],
            'v': ['b', 'g', 'h', 'j'],
            'w': ['q', 'e', 'r', 't'],
            'x': ['z', 'a', 's', 'd'],
            'y': ['t', 'u', 'i', 'o'],
            'z': ['x', 'c', 'v', 'b']
        }
        
        chars = list(text)
        num_typos = int(len(chars) * self.corruption_severity * 0.1)
        
        for _ in range(num_typos):
            idx = random.randint(0, len(chars) - 1)
            char = chars[idx].lower()
            if char in keyboard_neighbors:
                chars[idx] = random.choice(keyboard_neighbors[char])
        
        return ''.join(chars)


def get_sst2_loaders(batch_size: int, 
                     seed: int = 42,
                     val_split: float = 0.1,
                     tokenizer_name: str = 'bert-base-uncased',
                     max_length: int = 128) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Get SST-2 data loaders"""
    
    # Create datasets
    train_dataset = SST2Dataset(split='train', tokenizer_name=tokenizer_name, max_length=max_length)
    val_dataset = SST2Dataset(split='validation', tokenizer_name=tokenizer_name, max_length=max_length)
    test_dataset = SST2Dataset(split='validation', tokenizer_name=tokenizer_name, max_length=max_length)  # SST-2 uses validation as test
    
    # Create loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader