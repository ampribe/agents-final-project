import sys, json
from solver import solve
problem = {
    "key": b'\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10',
    "nonce": b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b',
    "plaintext": b'hello world',
    "associated_data": b'metadata'
}
result = solve(problem)
print('ciphertext length', len(result['ciphertext']))
print('tag length', len(result['tag']))
