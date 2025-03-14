import hashlib
import json
from time import time
from flask import Flask, jsonify, request
from urllib.parse import urlparse
import requests


class Block:
    def __init__(self, index, previous_hash, timestamp, data, hash):
        self.index = index
        self.previous_hash = previous_hash
        self.timestamp = timestamp
        self.data = data
        self.hash = hash


class Blockchain:
    def __init__(self):
        self.chain = []
        self.current_transactions = []
        self.nodes = set()
        self.new_block(previous_hash='1', proof=100)  # Create the genesis block

    def new_block(self, proof, previous_hash=None):
        block = Block(len(self.chain) + 1, previous_hash or self.hash(self.chain[-1]), time(), self.current_transactions, proof)
        self.current_transactions = []  # Reset the current list of transactions
        self.chain.append(block)
        return block

    def new_transaction(self, sender, recipient, amount):
        self.current_transactions.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })
        return self.last_block.index + 1

    @staticmethod
    def hash(block):
        block_string = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_string).hexdigest()

    @property
    def last_block(self):
        return self.chain[-1]

    def register_node(self, address):
        parsed_url = urlparse(address)
        self.nodes.add(parsed_url.netloc)

    def valid_chain(self, chain):
        last_block = chain[0]
        current_index = 1
        while current_index < len(chain):
            block = chain[current_index]
            if block.previous_hash != self.hash(last_block):
                return False
            last_block = block
            current_index += 1
        return True

    def resolve_conflicts(self):
        neighbors = self.nodes
        new_chain = None
        max_length = len(self.chain)

        for node in neighbors:
            response = requests.get(f'http://{node}/chain')
            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain
        if new_chain:
            self.chain = new_chain
            return True
        return False


app = Flask(__name__)
node_identifier = "Node-1"
blockchain = Blockchain()


@app.route('/mine', methods=['GET'])
def mine():
    last_block = blockchain.last_block
    proof = blockchain.proof_of_work(last_block)
    blockchain.new_transaction(sender="0", recipient=node_identifier, amount=1)
    block = blockchain.new_block(proof, last_block.hash)
    response = {
        'message': 'New Block Forged',
        'index': block.index,
        'transactions': block.data,
        'proof': block.hash,
        'previous_hash': block.previous_hash,
    }
    return jsonify(response), 200


@app.route('/transactions/new', methods=['POST'])
def new_transaction():
    values = request.get_json()
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400
    index = blockchain.new_transaction(values['sender'], values['recipient'], values['amount'])
    response = {'message': f'Transaction will be added to Block {index}'}
    return jsonify(response), 201


@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200


@app.route('/nodes/register', methods=['POST'])
def register_nodes():
    values = request.get_json()
    nodes = values.get('nodes')
    if nodes is None:
        return 'Error: Please supply a valid list of nodes', 400
    for node in nodes:
        blockchain.register_node(node)
    response = {'message': 'New nodes have been added', 'total_nodes': list(blockchain.nodes)}
    return jsonify(response), 201


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.resolve_conflicts()
    if replaced:
        response = {'message': 'Our chain was replaced', 'new_chain': blockchain.chain}
    else:
        response = {'message': 'Our chain is authoritative', 'chain': blockchain.chain}
    return jsonify(response), 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)