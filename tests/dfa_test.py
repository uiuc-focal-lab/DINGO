import unittest
import time, sys, os
import interegular
from syncode.mask_store.byte_fsm import ByteFSM
from transformers import AutoTokenizer
from typing import Dict, Tuple, Set, Optional, Any
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from dfa import compute_token_transitions


class TokenTransitionsTest(unittest.TestCase):
    """Test class for computing token transitions with different patterns."""
    
    def run_pattern_test(self, pattern, tokenizer_name="Qwen/Qwen2.5-1.5B"):
        """
        Run a test with the given pattern and tokenizer.
        
        Args:
            pattern: The regex pattern to test
            tokenizer_name: The name of the tokenizer to use
            
        Returns:
            Tuple of (num_states, vocab_size, total_time, num_finals)
        """
        print(f"Loading tokenizer from '{tokenizer_name}'...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        print(f"Creating ByteFSM for pattern: {pattern}")
        dfa = ByteFSM(pattern)
        
        # Get vocabulary size
        vocab_size = len(tokenizer.get_vocab())
        
        # Compute token transitions with timing
        print("Computing token transitions...")
        start_time = time.time()
        token_transitions, states = compute_token_transitions(tokenizer, dfa)
        end_time = time.time()
        total_time = end_time - start_time
        
        # Get number of states
        num_states = len(states)
        
        # Print the required information
        print(f"\nResults:")
        print(f"Number of states in the DFA: {num_states}")
        print(f"Number of vocabulary tokens: {vocab_size}")
        print(f"Total time taken: {total_time:.4f} seconds")
        
        # Group transitions by starting state and next state
        transitions_by_state = {}
        for (state, token_id), next_state in token_transitions.items():
            if next_state is not None:  # Only consider valid transitions
                if state not in transitions_by_state:
                    transitions_by_state[state] = {}
                if next_state not in transitions_by_state[state]:
                    transitions_by_state[state][next_state] = []
                transitions_by_state[state][next_state].append(token_id)
                
        # Display only core DFA information
        print(f"\nDFA Information:")
        print(f"  - States: {num_states}")
        print(f"  - Final states: {len(dfa.finals)}")
        
        # Show just a few transitions from the first 5 states
        print("\nSample transitions (first 5 states only):")
        states_shown = 0
        
        for state in sorted(transitions_by_state.keys()):
            if states_shown >= 5:
                break
                
            print(f"\nFrom state {state}:")
            transitions_shown = 0
            
            # Show just 3 destination states and 1 token for each
            for next_state, token_ids in sorted(transitions_by_state[state].items()):
                if transitions_shown >= 3 or not token_ids:
                    continue
                    
                token_id = token_ids[0]  # Just show the first token
                token_text = tokenizer.decode([token_id])
                is_final = " (FINAL)" if next_state in dfa.finals else ""
                print(f"  → State {next_state}{is_final} via token: '{token_text}'")
                transitions_shown += 1
                
            states_shown += 1
            
        return num_states, vocab_size, total_time, len(dfa.finals)
    
    def test_simple_pattern(self):
        """Test with a simple 'abc' pattern."""
        print("\n==== TESTING SIMPLE PATTERN ====")
        pattern = "abc"
        return self.run_pattern_test(pattern)
    
    def test_date_pattern(self):
        """Test with a date pattern in mm-dd-yyyy format."""
        print("\n==== TESTING DATE PATTERN ====")
        pattern = r"(0[1-9]|1[012])[-/\.](0[1-9]|[12][0-9]|3[01])[-/\.](19|20)\d\d"
        return self.run_pattern_test(pattern)
    
    def test_product_name_pattern(self):
        """Test with a product name pattern from a JSON schema."""
        print("\n==== TESTING PRODUCT NAME PATTERN ====")
        # Removed the ^ and $ anchors which are unsupported by interegular
        pattern = r"[A-Za-z0-9 \-_]+"
        return self.run_pattern_test(pattern)
    
    def test_gsm_symbolic(self):
        """Test with a pattern for a specific user JSON schema with multiple properties."""
        print("\n==== TESTING GSM SYMBOLIC ====")
        pattern = r'<<( )?(\(){0,3}( )?([a-z]|[0-9]{1,3})( )?(\)){0,3}(( )?([+\-]|\/\/|\/|%|\*|\*\*)( )?((\(){0,3}( )?)?([a-z]|[0-9]{1,3})( )?(\)){0,3})*( )?>>'

        return self.run_pattern_test(pattern)
    
    def test_user_json_schema(self):
        """Test with a pattern for a specific user JSON schema with multiple properties."""
        print("\n==== TESTING USER JSON SCHEMA ====")
        # This pattern matches a user profile JSON with specific properties:
        # - id: numeric ID
        # - name: string with letters and spaces
        # - email: email pattern
        # - active: boolean 
        # - role: one of several predefined roles
        pattern = r'\{"id":[0-9]+,"name":"[A-Za-z ]+","email":"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}","active":(true|false),"role":"(admin|user|guest)"\}'
        return self.run_pattern_test(pattern)
        
    def test_product_catalog_schema(self):
        """Test with a pattern for a product catalog entry JSON with nested structures."""
        print("\n==== TESTING PRODUCT CATALOG SCHEMA ====")
        # This pattern matches a product catalog JSON with these properties:
        # - sku: alphanumeric SKU code
        # - name: product name 
        # - price: decimal price
        # - category: product category
        # - inventory: numeric inventory count
        # - attributes: object with color and size
        pattern = r'\{"sku":"[A-Z0-9-]+","name":"[A-Za-z0-9 .,\-]+","price":[0-9]+\.[0-9]{2},"category":"[A-Za-z]+","inventory":[0-9]+,"attributes":\{"color":"[A-Za-z]+","size":"[A-Za-z0-9]+"}\}'
        return self.run_pattern_test(pattern)

    def test_whitespace(self):
        """Test with a pattern that matches whitespace characters."""
        print("\n==== TESTING WHITESPACE PATTERN ====")
        pattern = r" "
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B")
        dfa = ByteFSM(pattern)
        token_transitions, states = compute_token_transitions(tokenizer, dfa)
        ws_token = tokenizer.encode(" ")[0]
        assert token_transitions[(0, ws_token)] == 1


if __name__ == "__main__":
    # Create a single test instance
    test = TokenTransitionsTest()
    
    # Choose which patterns to test
    patterns_to_test = [
        # "test_simple_pattern",
        # "test_date_pattern",
        # "test_product_name_pattern",
        # "test_user_json_schema",
        # "test_product_catalog_schema",
        # "test_gsm_symbolic",
        "test_whitespace",
    ]
    
    results = []
    
    # Run selected tests
    for test_method_name in patterns_to_test:
        test_method = getattr(test, test_method_name)
        pattern_name = test_method_name.replace("test_", "").replace("_pattern", "")
        
        # Run the test and collect results
        num_states, vocab_size, time_taken, num_finals = test_method()
        results.append({
            "pattern": pattern_name,
            "states": num_states,
            "finals": num_finals,
            "time": time_taken
        })
    
    # Print summary
    print("\n==== SUMMARY ====")
    for result in results:
        print(f"Pattern: {result['pattern']}")
        print(f"  - States: {result['states']}")
        print(f"  - Final states: {result['finals']}")
        print(f"  - Time: {result['time']:.4f} seconds")
        print()
    
    print("All tests completed successfully!")
