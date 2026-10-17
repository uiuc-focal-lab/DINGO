import ast
import re
from typing import Dict, List, Optional, Set, Union

from lark import Lark


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
        self.max_replacement_depth = 100  # Limit recursion depth for safety

    def convert(self, grammar_text: str) -> Dict[str, str]:
        """
        Convert a Lark grammar string to regex patterns.

        Args:
            grammar_text: The Lark grammar as a string

        Returns:
            A dictionary mapping rule names to their regex patterns
        """
        self.rule_dict = {}
        grammar_text = grammar_text.rstrip() + "\n"
        # First, parse the grammar text to extract rules
        rules = self._extract_rules(grammar_text)

        # Check if the grammar is recursive
        if self._is_recursive(rules):
            raise ValueError("Recursive grammar detected. Cannot convert to regex.")

        # Parse the grammar with Lark's own parser
        try:
            parser = Lark(
                r"""
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
                STRING: /"(?:\\.|[^"\\])*"/ | /'(?:\\.|[^'\\])*'/
                REGEXP: /\/(?:\\.|[^\/\\])*\//
                
                %import common.WS
                %import common.NEWLINE -> _NEWLINE
                
                %ignore WS
            """,
                start="start",
            )

            tree = parser.parse(grammar_text)

            # Process each rule
            for rule_node in tree.children:
                if hasattr(rule_node, "data") and rule_node.data == "rule":
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
                raise ValueError(f"Invalid grammar: {e}") from e

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
        lines = grammar_text.strip().split("\n")
        rules = {}

        for line in lines:
            line = line.strip()
            if not line or ":" not in line:
                continue

            parts = line.split(":", 1)
            if len(parts) != 2:
                continue

            rule_name = parts[0].strip()
            definition = parts[1].strip()

            rules[rule_name] = definition

        return rules

    def _is_recursive(self, rules: Dict[str, str]) -> bool:
        # This converter expands references, so every cycle is unsupported,
        # even a regular left-recursive rule. Ignore names inside terminals.
        dependencies = {rule: set() for rule in rules}
        terminals = r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|/(?:\\.|[^/\\])*/'
        for name, definition in rules.items():
            references = re.sub(terminals, "", definition)
            dependencies[name] = {
                other for other in rules if re.search(r"\b" + re.escape(other) + r"\b", references)
            }
        return any(
            len(scc) > 1 or scc[0] in dependencies[scc[0]] for scc in self._find_sccs(dependencies)
        )

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
        if not hasattr(atom_node, "children") or len(atom_node.children) == 0:
            return ""

        # Process the term
        term_node = atom_node.children[0]
        term_regex = self._process_term(term_node)

        # Check if there's a modifier
        if len(atom_node.children) <= 1:
            return term_regex

        # Get the modifier and apply it
        modifier_node = atom_node.children[1]

        if hasattr(modifier_node, "data"):
            modifier_type = modifier_node.data
            if modifier_type == "optional":
                return f"(?:{term_regex})?"
            elif modifier_type == "zero_or_more":
                return f"(?:{term_regex})*"
            elif modifier_type == "one_or_more":
                return f"(?:{term_regex})+"

        # Default - no modifier applied
        return term_regex

    def _process_term(self, term_node) -> str:
        """Process a term (rule reference, string, regexp or group)."""
        if not hasattr(term_node, "data"):
            return ""

        if term_node.data == "rule_ref":
            rule_name = term_node.children[0].value
            # Use a placeholder that will be resolved later
            return f"__RULE_REF_{rule_name}__"

        elif term_node.data == "string":
            # Convert string literal to regex
            string_value = term_node.children[0].value
            # Remove quotes
            string_content = ast.literal_eval(string_value)
            # Escape regex special characters
            escaped = re.escape(string_content)
            return escaped

        elif term_node.data == "regexp":
            # Extract regexp content
            regexp_value = term_node.children[0].value
            # Remove the surrounding slashes
            return "(?:" + regexp_value[1:-1] + ")"

        elif term_node.data == "group":
            # Process the group's expansions
            return "(?:" + self._process_expansions(term_node.children[0]) + ")"

        return ""  # Fallback

    def _resolve_rule_references(self):
        """Replace rule reference placeholders with actual regex patterns in a memory-efficient way."""
        for pattern in self.rule_dict.values():
            for ref in re.findall(r"__RULE_REF_([a-z][a-z0-9_]*?)__", pattern):
                if ref not in self.rule_dict:
                    raise ValueError(f"Undefined rule reference: {ref}")
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

                        pattern = pattern.replace(placeholder, replacement)
                        modified = True
                        unresolved = True

                if modified:
                    self.rule_dict[rule_name] = pattern

        # Check if we still have unresolved references
        for pattern in self.rule_dict.values():
            for ref_name in self.rule_dict:
                placeholder = f"__RULE_REF_{ref_name}__"
                if placeholder in pattern:
                    raise ValueError(
                        f"Could not resolve all rule references after {max_iterations} iterations. The grammar might be too complex or recursive."
                    )


def lark_to_regex(
    grammar_text: str, start_rule: Optional[str] = None
) -> Union[Dict[str, str], str]:
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
