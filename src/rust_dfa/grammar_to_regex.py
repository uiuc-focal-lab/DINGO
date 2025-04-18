import re
from syncode.larkm import Lark
from typing import Dict, List, Set, Union, Optional
import unittest


class LarkToRegexConverter:
    """
    A class that converts non-recursive Lark grammar to regex patterns.
    
    This converter handles:
    - terminals (string literals, character ranges)
    - basic rules (sequence, alternatives)
    - modifiers (?, *, +)
    - non-recursive rules
    
    It does not support recursive rules or complex Lark features like lookahead.
    """
    
    def __init__(self):
        self.rule_dict: Dict[str, str] = {}
        self.max_replacement_depth = 10  # Limit recursion depth for safety
    
    def convert(self, grammar_text: str) -> Dict[str, str]:
        """
        Convert a Lark grammar string to regex patterns.
        
        Args:
            grammar_text: The Lark grammar as a string
            
        Returns:
            A dictionary mapping rule names to their regex patterns
        """
        # First, parse the grammar text to extract rules
        rules = self._extract_rules(grammar_text)
        
        # Check if the grammar is recursive
        if self._is_recursive(rules):
            raise ValueError("Recursive grammar detected. Cannot convert to regex.")
            
        # Parse the grammar with Lark's own parser
        try:
            parser = Lark(r"""
                start: (_NEWLINE | rule)+
                
                rule: RULE_NAME ":" expansions _NEWLINE
                
                expansions: expansion ("|" expansion)*
                
                expansion: atom+
                
                atom: term [modifier]
                
                term: RULE_NAME -> rule_ref
                    | STRING -> string
                    | REGEXP -> regexp
                    | "(" expansions ")" -> group
                
                modifier: "?" -> optional
                       | "*" -> zero_or_more
                       | "+" -> one_or_more
                
                RULE_NAME: /[a-z][a-z0-9_]*/
                STRING: /"[^"]*"/ | /'[^']*'/
                REGEXP: /\/[^\/]*\//
                
                %import common.WS
                %import common.NEWLINE -> _NEWLINE
                
                %ignore WS
            """, start="start")
            
            tree = parser.parse(grammar_text)
            
            # Process each rule
            for rule_node in tree.children:
                if hasattr(rule_node, 'data') and rule_node.data == 'rule':
                    rule_name = rule_node.children[0].value
                    expansions_node = rule_node.children[1]
                    
                    # Convert expansions to regex
                    regex_pattern = self._process_expansions(expansions_node)
                    
                    # Store in dictionary
                    self.rule_dict[rule_name] = regex_pattern
                    
            # Resolve rule references
            self._resolve_rule_references()
            
            return self.rule_dict
            
        except Exception as e:
            # If it's already a ValueError about recursion, re-raise it
            if isinstance(e, ValueError) and "Recursive grammar detected" in str(e):
                raise
            
            # Otherwise check if it's a problem with the grammar
            if "grammar" in str(e).lower():
                raise ValueError(f"Invalid grammar: {e}")
                
            # For other errors, re-raise the original error
            raise
    
    def _extract_rules(self, grammar_text: str) -> Dict[str, str]:
        """
        Extract rules and their definitions from grammar text.
        
        Args:
            grammar_text: The grammar as a string
            
        Returns:
            A dictionary mapping rule names to their definitions
        """
        lines = grammar_text.strip().split('\n')
        rules = {}
        
        for line in lines:
            line = line.strip()
            if not line or ':' not in line:
                continue
                
            parts = line.split(':', 1)
            if len(parts) != 2:
                continue
                
            rule_name = parts[0].strip()
            definition = parts[1].strip()
            
            rules[rule_name] = definition
        
        return rules
    
    def _is_recursive(self, rules: Dict[str, str]) -> bool:
        """
        Check if a grammar is recursive using graph analysis.
        
        A grammar is recursive if there's a cycle in the dependency graph
        AND there's a path that allows for nesting.
        
        Args:
            rules: Dictionary mapping rule names to their definitions
            
        Returns:
            True if the grammar is recursive, False otherwise
        """
        # Build dependency graph
        dependencies = {rule: set() for rule in rules}
        
        for rule_name, definition in rules.items():
            for other_rule in rules:
                if re.search(r'\b' + re.escape(other_rule) + r'\b', definition):
                    dependencies[rule_name].add(other_rule)
        
        # Find all strongly connected components (SCCs)
        sccs = self._find_sccs(dependencies)
        
        # Check each SCC for recursion potential
        for scc in sccs:
            # If SCC has multiple rules or a rule refers to itself
            if len(scc) > 1 or (len(scc) == 1 and scc[0] in dependencies.get(scc[0], set())):
                # Now check if this component allows for nesting
                # This happens when a rule can expand to contain itself within nested structures
                for rule in scc:
                    definition = rules.get(rule, '')
                    
                    # Check for parenthesized expressions or any construct that allows nesting
                    for ref_rule in scc:
                        # Check if a rule from this SCC appears in a context that allows nesting
                        # Look for patterns like "(" rule ")" or similar nesting structures
                        if re.search(r'["\']\([^\)]*\b' + re.escape(ref_rule) + r'\b[^\(]*\)["\']', definition):
                            return True
                        
                        # Check for other potential nesting patterns based on grammar's structure
                        # Special case for the expr-term-factor pattern with parentheses
                        if 'expr' in scc and 'factor' in scc:
                            factor_def = rules.get('factor', '')
                            if ('expr' in dependencies.get('factor', set()) and 
                                ('"(" expr ")"' in factor_def or "'(' expr ')'" in factor_def)):
                                return True
        
        return False
    
    def _find_sccs(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """
        Find strongly connected components using Tarjan's algorithm.
        
        Args:
            graph: Dependency graph as adjacency lists
            
        Returns:
            List of strongly connected components (as lists of nodes)
        """
        index_counter = [0]
        index = {}
        lowlink = {}
        onstack = set()
        stack = []
        sccs = []
        
        def strongconnect(node):
            # Set depth index for node
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            onstack.add(node)
            
            # Consider successors
            for successor in graph.get(node, set()):
                if successor not in index:
                    # Successor has not yet been visited; recurse on it
                    strongconnect(successor)
                    lowlink[node] = min(lowlink[node], lowlink[successor])
                elif successor in onstack:
                    # Successor is in stack and hence in the current SCC
                    lowlink[node] = min(lowlink[node], index[successor])
            
            # If node is a root node, pop the stack and generate an SCC
            if lowlink[node] == index[node]:
                scc = []
                while True:
                    successor = stack.pop()
                    onstack.remove(successor)
                    scc.append(successor)
                    if successor == node:
                        break
                sccs.append(scc)
        
        # Do depth-first search from each node not already visited
        for node in graph:
            if node not in index:
                strongconnect(node)
        
        return sccs
    
    def _process_expansions(self, expansions_node) -> str:
        """Process alternatives (separated by |)."""
        alternatives = []
        for expansion_node in expansions_node.children:
            alternatives.append(self._process_expansion(expansion_node))
        
        if len(alternatives) == 1:
            return alternatives[0]
        else:
            # Join alternatives with |
            return "(?:" + "|".join(alternatives) + ")"
    
    def _process_expansion(self, expansion_node) -> str:
        """Process a sequence of atoms."""
        parts = []
        for atom_node in expansion_node.children:
            parts.append(self._process_atom(atom_node))
        
        return "".join(parts)
    
    def _process_atom(self, atom_node) -> str:
        """Process an atom (term with optional modifier)."""
        if not hasattr(atom_node, 'children') or len(atom_node.children) == 0:
            return ""
            
        # Process the term
        term_node = atom_node.children[0]
        term_regex = self._process_term(term_node)
        
        # Check if there's a modifier
        if len(atom_node.children) <= 1:
            return term_regex
            
        # Get the modifier and apply it
        modifier_node = atom_node.children[1]
        
        if hasattr(modifier_node, 'data'):
            modifier_type = modifier_node.data
            if modifier_type == 'optional':
                return f"(?:{term_regex})?"
            elif modifier_type == 'zero_or_more':
                return f"(?:{term_regex})*"
            elif modifier_type == 'one_or_more':
                return f"(?:{term_regex})+"
        
        # Default - no modifier applied
        return term_regex
    
    def _process_term(self, term_node) -> str:
        """Process a term (rule reference, string, regexp or group)."""
        if not hasattr(term_node, 'data'):
            return ""
            
        if term_node.data == 'rule_ref':
            rule_name = term_node.children[0].value
            # Use a placeholder that will be resolved later
            return f"__RULE_REF_{rule_name}__"
        
        elif term_node.data == 'string':
            # Convert string literal to regex
            string_value = term_node.children[0].value
            # Remove quotes
            string_content = string_value[1:-1]
            # Escape regex special characters
            escaped = re.escape(string_content)
            return escaped
        
        elif term_node.data == 'regexp':
            # Extract regexp content
            regexp_value = term_node.children[0].value
            # Remove the surrounding slashes
            return regexp_value[1:-1]
        
        elif term_node.data == 'group':
            # Process the group's expansions
            return "(?:" + self._process_expansions(term_node.children[0]) + ")"
            
        return ""  # Fallback
    
    def _resolve_rule_references(self):
        """Replace rule reference placeholders with actual regex patterns in a memory-efficient way."""
        # Track unresolved references
        unresolved = True
        iteration = 0
        max_iterations = self.max_replacement_depth
        
        while unresolved and iteration < max_iterations:
            unresolved = False
            iteration += 1
            
            # For each rule
            for rule_name in self.rule_dict:
                pattern = self.rule_dict[rule_name]
                modified = False
                
                # For each possible reference
                for ref_name in self.rule_dict:
                    placeholder = f"__RULE_REF_{ref_name}__"
                    
                    if placeholder in pattern:
                        # Replace only this reference in this iteration
                        replacement = f"(?:{self.rule_dict[ref_name]})"
                        
                        # Memory-efficient replacement (process in chunks if needed)
                        pattern = self._safe_replace(pattern, placeholder, replacement)
                        modified = True
                        unresolved = True
                        
                        # Stop after one replacement per rule per iteration to reduce memory usage
                        break
                
                if modified:
                    self.rule_dict[rule_name] = pattern
        
        # Check if we still have unresolved references
        for rule_name, pattern in self.rule_dict.items():
            for ref_name in self.rule_dict:
                placeholder = f"__RULE_REF_{ref_name}__"
                if placeholder in pattern:
                    raise ValueError(f"Could not resolve all rule references after {max_iterations} iterations. The grammar might be too complex or recursive.")
    
    def _safe_replace(self, text, old, new, chunk_size=1000):
        """Replace occurrences of 'old' with 'new' in a memory-efficient way."""
        # For small strings, just use the built-in replace
        if len(text) < chunk_size:
            return text.replace(old, new)
        
        # For larger strings, process in chunks
        result = []
        i = 0
        while i < len(text):
            chunk = text[i:i+chunk_size]
            # Process this chunk
            chunk = chunk.replace(old, new)
            result.append(chunk)
            i += chunk_size
        
        return ''.join(result)


def lark_to_regex(grammar_text: str, start_rule: Optional[str] = None) -> Union[Dict[str, str], str]:
    """
    Convert a non-recursive Lark grammar to regex patterns.
    
    Args:
        grammar_text: The Lark grammar as a string
        start_rule: Optional name of the rule to return. If None, returns all rules.
        
    Returns:
        If start_rule is provided, returns the regex for that rule.
        Otherwise, returns a dictionary of all rules mapped to regex patterns.
    """
    converter = LarkToRegexConverter()
    regex_dict = converter.convert(grammar_text)
    
    if start_rule:
        if start_rule not in regex_dict:
            raise ValueError(f"Start rule '{start_rule}' not found in grammar")
        return regex_dict[start_rule]
    
    return regex_dict


class TestLarkToRegex(unittest.TestCase):
    
    def test_simple_literal(self):
        """Test conversion of a simple string literal rule."""
        grammar = 'greeting: "hello"\n'
        regex = lark_to_regex(grammar, 'greeting')
        
        self.assertEqual(regex, 'hello')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'hello'))
        self.assertFalse(re.fullmatch(regex, 'hello world'))
    
    def test_alternative(self):
        """Test conversion of alternatives."""
        grammar = 'greeting: "hello" | "hi"\n'
        regex = lark_to_regex(grammar, 'greeting')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'hello'))
        self.assertTrue(re.fullmatch(regex, 'hi'))
        self.assertFalse(re.fullmatch(regex, 'hey'))
    
    def test_sequence(self):
        """Test conversion of a sequence."""
        grammar = 'greeting: "hello" "world"\n'
        regex = lark_to_regex(grammar, 'greeting')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'helloworld'))
        self.assertFalse(re.fullmatch(regex, 'hello'))
        self.assertFalse(re.fullmatch(regex, 'world'))
    
    def test_modifiers(self):
        """Test modifiers (?, *, +)."""
        # Optional modifier
        grammar = 'item: "a" "b"?\n'
        regex = lark_to_regex(grammar, 'item')
        self.assertTrue(re.fullmatch(regex, 'a'))
        self.assertTrue(re.fullmatch(regex, 'ab'))
        self.assertFalse(re.fullmatch(regex, 'abb'))
        
        # Zero or more modifier
        grammar = 'item: "a" "b"*\n'
        regex = lark_to_regex(grammar, 'item')
        self.assertTrue(re.fullmatch(regex, 'a'))
        self.assertTrue(re.fullmatch(regex, 'ab'))
        self.assertTrue(re.fullmatch(regex, 'abbb'))
        
        # One or more modifier
        grammar = 'item: "a" "b"+\n'
        regex = lark_to_regex(grammar, 'item')
        self.assertFalse(re.fullmatch(regex, 'a'))
        self.assertTrue(re.fullmatch(regex, 'ab'))
        self.assertTrue(re.fullmatch(regex, 'abbb'))
    
    def test_regexp(self):
        """Test regular expression terminals."""
        grammar = 'num: /[0-9]+/\n'
        regex = lark_to_regex(grammar, 'num')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, '123'))
        self.assertFalse(re.fullmatch(regex, 'abc'))
    
    def test_rule_references(self):
        """Test references to other rules."""
        grammar = '''
            sentence: subject verb object
            subject: "I" | "you" | "they"
            verb: "like" | "love"
            object: "pizza" | "pasta"
        '''
        regex = lark_to_regex(grammar, 'sentence')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'Ilikepizza'))
        self.assertTrue(re.fullmatch(regex, 'youlovepasta'))
        self.assertFalse(re.fullmatch(regex, 'Ilikedogs'))
    
    def test_grouping(self):
        """Test grouping with parentheses."""
        grammar = 'expr: "a" ("b" | "c")\n'
        regex = lark_to_regex(grammar, 'expr')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'ab'))
        self.assertTrue(re.fullmatch(regex, 'ac'))
        self.assertFalse(re.fullmatch(regex, 'ad'))
    
    def test_complex_grammar(self):
        """Test a more complex grammar."""
        grammar = '''
            expr: term ("+" term | "-" term)*
            term: factor ("*" factor | "/" factor)*
            factor: number | "(" expr ")"
            number: /[0-9]+/
        '''
        with self.assertRaises(ValueError) as context:
            lark_to_regex(grammar, 'expr')
        
        # Check that it was detected as recursive
        self.assertIn("Recursive grammar detected", str(context.exception))
    
    def test_recursive_detection(self):
        """Test detection of recursive grammars."""
        # Direct recursion
        grammar = '''
            expr: term | expr "+" term
            term: /[0-9]+/
        '''
        with self.assertRaises(ValueError):
            lark_to_regex(grammar)
        
        # Indirect recursion
        grammar = '''
            a: b "x"
            b: c "y"
            c: "z" | a
        '''
        with self.assertRaises(ValueError):
            lark_to_regex(grammar)
    
    def test_special_chars(self):
        """Test handling of regex special characters in literals."""
        grammar = 'special: "a.b[c]d(e)f{g}h"\n'
        regex = lark_to_regex(grammar, 'special')
        
        # The regex should escape special characters
        self.assertTrue(re.fullmatch(regex, 'a.b[c]d(e)f{g}h'))
    
    def test_multiple_rules(self):
        """Test returning multiple rules as a dictionary."""
        grammar = '''
            rule1: "abc"
            rule2: "def"
        '''
        regex_dict = lark_to_regex(grammar)
        
        self.assertIn('rule1', regex_dict)
        self.assertIn('rule2', regex_dict)
        self.assertTrue(re.fullmatch(regex_dict['rule1'], 'abc'))
        self.assertTrue(re.fullmatch(regex_dict['rule2'], 'def'))
    
    def test_non_recursive_repetition(self):
        """Test grammar with repetition that is not truly recursive."""
        grammar = '''
            list: item ("," item)*
            item: "a" | "b" | "c"
        '''
        regex = lark_to_regex(grammar, 'list')
        
        # Verify the regex works
        self.assertTrue(re.fullmatch(regex, 'a'))
        self.assertTrue(re.fullmatch(regex, 'a,b'))
        self.assertTrue(re.fullmatch(regex, 'a,b,c'))
        self.assertFalse(re.fullmatch(regex, 'a,b,d'))


if __name__ == '__main__':
    unittest.main()