# Active Challenges — SAGE Enigma Machine

## Challenge 1: Quantum Error Correction Code Improvement
**ID:** quantum-ecc-001  
**Description:**  
Design a novel quantum error correction code for surface codes that outperforms current methods in logical error rate under realistic noise models.

**Verification Spec:**  
```yaml
verification:
  type: quantum_simulation
  requirements:
    - logical_error_rate < 1e-4 at physical_error_rate=0.01
    - circuit_depth <= 20
    - qubit_count <= 49
    - must_pass_stabilizer_check: true
    - must_be_deterministic_under_noise: true
  test_cases:
    - input: "surface_code_3x3"
      expected: "logical_error_rate <= 5e-5"
  scoring_weights:
    fidelity: 0.4
    efficiency: 0.3
    novelty: 0.2
    verifier_compliance: 0.1
