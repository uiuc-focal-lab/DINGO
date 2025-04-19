// src/lib.rs
use pyo3::prelude::*;
use regex_automata::dfa::dense;
use regex_automata::dfa::Automaton;
use regex_automata::util::primitives::StateID;
use regex_automata::util::start;
use regex_automata::Anchored;

#[pyclass]
struct RegexDFA {
    dfa: Option<dense::DFA<Vec<u32>>>,
    size: usize,
    alive_states: std::collections::HashSet<usize>,
}

#[pymethods]
impl RegexDFA {
    #[new]
    fn new() -> Self {
        RegexDFA {
            dfa: None,
            size: 0,
            alive_states: std::collections::HashSet::new(),
        }
    }

    fn initialize(&mut self, regex_str: &str) -> PyResult<usize> {
        // Create a DFA directly from the regex with minimization enabled
        let dfa = match dense::Builder::new()
            .configure(dense::Config::new().minimize(true))
            .build(regex_str) {
                Ok(dfa) => dfa,
                Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Failed to build DFA: {}", e)
                ))
            };
        
        // print the regex and the size of the DFA
        println!("Regex: {}", regex_str);

        self.size = dfa.memory_usage();
        
        // Store the DFA
        self.dfa = Some(dfa);
        
        // Compute alive states
        self.compute_alive_states()?;
        
        let cnt_states = self.count_reachable_states()?;
        println!("Reachable states: {}", cnt_states);

        Ok(cnt_states)
    }
    
    fn get_size(&self) -> usize {
        self.size
    }

    fn walk(&self, input: &str) -> PyResult<usize> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Convert string to bytes
        let bytes = input.as_bytes();
        
        // Set up the config with anchored mode
        let config = start::Config::new().anchored(Anchored::Yes);
        
        // Get the start state
        let mut state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Walk through the DFA state by state
        for &byte in bytes {
            state = dfa.next_state(state, byte);
            
            if dfa.is_dead_state(state) {
                break;
            }
        }
        
        // Return the state ID as a usize
        Ok(state.as_usize())
    }
    
    fn matches(&self, input: &str) -> PyResult<bool> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Get the final state by walking the DFA
        let final_state_usize = self.walk(input)?;
        
        // Convert usize to StateID
        let final_state = StateID::new(final_state_usize)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Invalid state ID: {}", e)
            ))?;
        
        // Check if the final state is a match state after EOI
        let eoi_state = dfa.next_eoi_state(final_state);
        
        // Return whether it's a match state
        Ok(dfa.is_match_state(eoi_state))
    }

    fn count_reachable_states(&self) -> PyResult<usize> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Start from the initial state
        let config = start::Config::new();
        let start_state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Use BFS to traverse the DFA
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Add the start state to the queue
        queue.push_back(start_state);
        visited.insert(start_state.as_usize());
        
        // Process all reachable states
        while let Some(current_state) = queue.pop_front() {
            // For each possible input byte (0-255)
            for byte in 0..=255u8 {
                let next_state = dfa.next_state(current_state, byte);
                
                // Skip dead states
                if dfa.is_dead_state(next_state) {
                    continue;
                }
                
                // If we haven't visited this state yet
                if visited.insert(next_state.as_usize()) {
                    queue.push_back(next_state);
                }
            }
        }
        
        // Return the count of reachable states
        println!("All reachable states: {:?}", visited);
        Ok(visited.len())
    }

    fn compute_alive_states(&mut self) -> PyResult<()> {
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        self.alive_states.clear();
        
        // First, find all accepting states
        let mut queue = std::collections::VecDeque::new();
        
        // Get all reachable states
        let reachable_states = self.get_reachable_states()?;
        
        // Add all accepting states to the queue and mark as alive
        for &state_id in &reachable_states {
            let state = StateID::new(state_id)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    format!("Invalid state ID: {}", e)
                ))?;
                
            let eoi_state = dfa.next_eoi_state(state);
            if dfa.is_match_state(eoi_state) {
                queue.push_back(state_id);
                self.alive_states.insert(state_id);
            }
        }
        
        // Build a reverse transition map
        let mut reverse_transitions: std::collections::HashMap<usize, Vec<usize>> = 
            std::collections::HashMap::new();
            
        for &from_id in &reachable_states {
            if dfa.is_dead_state(StateID::new(from_id).unwrap()) {
                continue;
            }
            
            let from = StateID::new(from_id).unwrap();
            
            for byte in 0..=255u8 {
                let to = dfa.next_state(from, byte);
                let to_id = to.as_usize();
                
                if dfa.is_dead_state(to) {
                    continue;
                }
                
                reverse_transitions.entry(to_id)
                    .or_insert_with(Vec::new)
                    .push(from_id);
            }
        }
        
        // Perform backward search from accepting states
        while let Some(to_id) = queue.pop_front() {
            if let Some(from_ids) = reverse_transitions.get(&to_id) {
                for &from_id in from_ids {
                    if self.alive_states.insert(from_id) {
                        // If this is a new alive state, add it to the queue
                        queue.push_back(from_id);
                    }
                }
            }
        }
        
        println!("Alive states: {} out of {}", self.alive_states.len(), reachable_states.len());
        Ok(())
    }
    
    fn get_reachable_states(&self) -> PyResult<std::collections::HashSet<usize>> {
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Start from the initial state
        let config = start::Config::new();
        let start_state = match dfa.start_state(&config) {
            Ok(state) => state,
            Err(e) => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                format!("Failed to get start state: {}", e)
            ))
        };
        
        // Use BFS to traverse the DFA
        let mut visited = std::collections::HashSet::new();
        let mut queue = std::collections::VecDeque::new();
        
        // Add the start state to the queue
        queue.push_back(start_state);
        visited.insert(start_state.as_usize());
        
        // Process all reachable states
        while let Some(current_state) = queue.pop_front() {
            // For each possible input byte (0-255)
            for byte in 0..=255u8 {
                let next_state = dfa.next_state(current_state, byte);
                
                // Skip dead states
                if dfa.is_dead_state(next_state) {
                    continue;
                }
                
                // If we haven't visited this state yet
                if visited.insert(next_state.as_usize()) {
                    queue.push_back(next_state);
                }
            }
        }
        
        Ok(visited)
    }
    
    fn prefix_matches(&self, input: &str) -> PyResult<bool> {
        // Ensure DFA is initialized
        let dfa = match &self.dfa {
            Some(dfa) => dfa,
            None => return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DFA not initialized. Call initialize() first."
            ))
        };
        
        // Get the final state by walking the DFA
        let final_state_usize = self.walk(input)?;
        
        // If it's a dead state, it can't lead to a match
        if dfa.is_dead_state(StateID::new(final_state_usize).unwrap()) {
            return Ok(false);
        }
        
        // Check if the state is alive (can reach an accepting state)
        Ok(self.alive_states.contains(&final_state_usize))
    }
}

#[pymodule]
fn rust_dfa(_py: Python, m: &PyModule) -> PyResult<()> {  // This must match the module name
    m.add_class::<RegexDFA>()?;
    Ok(())
}