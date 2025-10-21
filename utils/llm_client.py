"""
LLM Client wrapper for OpenAI and Anthropic APIs
"""
import os
import json
import re
from typing import Dict, Any, Optional
import config

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI not installed. Run: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ Anthropic not installed. Run: pip install anthropic")


class LLMClient:
    """
    Unified client for LLM API calls (OpenAI and Anthropic)
    """
    
    def __init__(self, provider: str = None):
        """
        Initialize LLM client
        
        Args:
            provider: "openai" or "anthropic" (default from config)
        """
        self.provider = provider or config.LLM_PROVIDER
        
        if self.provider == "openai":
            self._init_openai()
        elif self.provider == "anthropic":
            self._init_anthropic()
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")
    
    def _init_openai(self):
        """Initialize OpenAI client"""
        if not OPENAI_AVAILABLE:
            raise ImportError(
                "OpenAI package not installed. "
                "Install with: pip install openai"
            )
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable not set. "
                "Set it with: export OPENAI_API_KEY='your-key-here'"
            )
        
        # Initialize OpenAI client (new API)
        self.client = OpenAI(api_key=self.api_key)
        self.model = config.OPENAI_MODEL
        print(f"✓ OpenAI client initialized (model: {self.model})")
    
    def _init_anthropic(self):
        """Initialize Anthropic client"""
        if not ANTHROPIC_AVAILABLE:
            raise ImportError(
                "Anthropic package not installed. "
                "Install with: pip install anthropic"
            )
        
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError(
                "ANTHROPIC_API_KEY environment variable not set. "
                "Set it with: export ANTHROPIC_API_KEY='your-key-here'"
            )
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        self.model = config.ANTHROPIC_MODEL
        print(f"✓ Anthropic client initialized (model: {self.model})")
    
    def call(self, system_prompt: str, user_prompt: str, 
             temperature: float = None, max_tokens: int = 2000) -> str:
        """
        Make an LLM API call
        
        Args:
            system_prompt: System instructions for the LLM
            user_prompt: User message/query
            temperature: Sampling temperature (default from config)
            max_tokens: Maximum tokens in response
            
        Returns:
            LLM response text
        """
        temp = temperature if temperature is not None else config.TEMPERATURE
        
        if self.provider == "openai":
            return self._call_openai(system_prompt, user_prompt, temp, max_tokens)
        elif self.provider == "anthropic":
            return self._call_anthropic(system_prompt, user_prompt, temp, max_tokens)
    
    def _call_openai(self, system_prompt: str, user_prompt: str, 
                     temperature: float, max_tokens: int) -> str:
        """Call OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content.strip()
        
        except Exception as e:
            raise RuntimeError(f"OpenAI API call failed: {e}")
    
    def _call_anthropic(self, system_prompt: str, user_prompt: str, 
                       temperature: float, max_tokens: int) -> str:
        """Call Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ]
            )
            return response.content[0].text.strip()
        
        except Exception as e:
            raise RuntimeError(f"Anthropic API call failed: {e}")
    
    def call_with_json_response(self, system_prompt: str, user_prompt: str, 
                                temperature: float = None, 
                                max_tokens: int = 2000) -> Dict[str, Any]:
        """
        Make an LLM call expecting JSON response
        
        Args:
            system_prompt: System instructions
            user_prompt: User message
            temperature: Sampling temperature
            max_tokens: Maximum tokens
            
        Returns:
            Parsed JSON dictionary
        """
        # Add JSON instruction to system prompt
        enhanced_system = system_prompt + "\n\nIMPORTANT: You MUST respond with valid JSON only. No additional text or explanation."
        
        # Get response
        response_text = self.call(enhanced_system, user_prompt, temperature, max_tokens)
        
        # Parse JSON from response
        return self._parse_json_response(response_text)
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response, handling markdown code blocks
        
        Args:
            response_text: Raw response from LLM
            
        Returns:
            Parsed JSON dictionary
        """
        # Try to extract JSON from markdown code blocks using regex
        # Pattern 1: ``````
        pattern1 = r'``````'
        match = re.search(pattern1, response_text, re.DOTALL)
        if match:
            response_text = match.group(1).strip()
        else:
            # Pattern 2: ``````
            pattern2 = r'``````'
            match = re.search(pattern2, response_text, re.DOTALL)
            if match:
                response_text = match.group(1).strip()
        
        # Try to parse JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError as e:
            # If parsing fails, try to extract JSON object from text
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group(0))
                except:
                    pass
            
            raise ValueError(
                f"Failed to parse JSON from LLM response.\n"
                f"Error: {e}\n"
                f"Response preview: {response_text[:500]}..."
            )
    
    def test_connection(self) -> bool:
        """
        Test if the LLM connection is working
        
        Returns:
            True if connection successful
        """
        try:
            response = self.call(
                system_prompt="You are a helpful assistant.",
                user_prompt="Say 'Hello'",
                temperature=0,
                max_tokens=10
            )
            return len(response) > 0
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False
