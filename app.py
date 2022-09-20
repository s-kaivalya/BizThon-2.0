import pandas as pd
import pandas.testing as tm
import numpy as np
from numpy import loadtxt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import xgboost
from xgboost import XGBClassifier
import hashlib
import json
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import requests
from flask import Flask, jsonify, request

#concatenate data into a single data frame

account= pd.read_csv("account.csv")
order= pd.read_csv("order.csv")
transaction= pd.read_csv("transaction.csv")

X= pd.concat([account,order,transaction], axis=0)

#dividing the data into train and test sets for the k-means model

X_new= X.copy() #create a copy of your data 

x_train = X_new.sample(frac=0.40, random_state=0)
x_test = X_new.drop(x_train.index)

#Create a class to store the block chain

class Blockchain:
    def __init__(self):
        self.current_trans = []
        self.chain = []
        self.nodes = set()

        #Create the genesis block
        self.new_block(prev_hash='1', proof=100)

    def new_node(self, address):
        """
        Add a new node. View the node here:'http://192.168.0.5:5000'
        """

        parsed_url = urlparse(address)
        if parsed_url.netloc:
            self.nodes.add(parsed_url.netloc)
        elif parsed_url.path:
            self.nodes.add(parsed_url.path)
        else:
            raise ValueError('Invalid URL. Please try again.')


    def valid_chain(self, chain):
        """
        Determine if blockchain is valid.
        """

        prev_block = chain[0]
        current_index = 1

        while current_index < len(chain):
            block = chain[current_index]
            print(f'{prev_block}')
            print(f'{block}')
            print("\n-----------\n")
            #Check that the hash of the block is correct
            prev_block_hash = self.hash(prev_block)
            if block['prev_hash'] != prev_block_hash:
                return False

            #Check that the Proof of Work is correct
            if not self.valid_proof(prev_block['proof'], block['proof'], prev_block_hash):
                return False

            prev_block = block
            current_index += 1

        return True

    def conflict_resolution(self):
        """
        Resolves conflicts by replacing current chain with the longest one in the network.
        """

        neighbours = self.nodes
        new_chain = None

        #Identifying long chains
        max_length = len(self.chain)

        #Grab and verify the chains from all the nodes in the network
        for node in neighbours:
            response = requests.get(f'http://{node}/chain')

            if response.status_code == 200:
                length = response.json()['length']
                chain = response.json()['chain']

                #Check if the length is longer and the chain is valid
                if length > max_length and self.valid_chain(chain):
                    max_length = length
                    new_chain = chain

        #Replace chain if a valid longer chain is discovered
        if new_chain:
            self.chain = new_chain
            return True

        return False

    def new_block(self, proof, prev_hash):

        block = {
            'index': len(self.chain) + 1,
            'timestamp': time(),
            'transactions': self.current_trans,
            'proof': proof,
            'prev_hash': prev_hash or self.hash(self.chain[-1]),
        }

        #Reset the current list of transactions
        self.current_trans = []

        self.chain.append(block)
        return block

    def new_trans(self, sender, recipient, amount):
        """
        Creates a new transaction to go into the next mined Block.
        """
        self.current_trans.append({
            'sender': sender,
            'recipient': recipient,
            'amount': amount,
        })

        return self.prev_block['index'] + 1

    @property
    def prev_block(self):
        return self.chain[-1]

    @staticmethod
    def hash(block):
        """
        SHA-256 encryption
        """

        #Ensure that dictionary is ordered, to avoid inconsistent hashes.
        block_str = json.dumps(block, sort_keys=True).encode()
        return hashlib.sha256(block_str).hexdigest()

    def proof_of_work(self, prev_block):
        
         #Proof of Work Algorithm:
         #- Find a number p' such that hash(pp') contains leading 4 zeroes
         #- Where p is the previous proof, and p' is the new proof

        prev_proof = prev_block['proof']
        prev_hash = self.hash(prev_block)

        proof = 0
        while self.valid_proof(prev_proof, proof, prev_hash) is False:
            proof += 1

        return proof

    @staticmethod
    def valid_proof(prev_proof, proof, prev_hash):

        #Validates Proof

        guess = f'{prev_proof}{proof}{prev_hash}'.encode()
        guess_hash = hashlib.sha256(guess).hexdigest()
        return guess_hash[:4] == "0000"



#Instantiate the Node
app = Flask(__name__)

#Generate a globally unique address for this node
node_id = str(uuid4()).replace('-', '')

#Instantiate the Blockchain
blockchain = Blockchain()


@app.route('/mine', methods=['GET'])
def mine():
    #Run the proof of work algorithm to get the next proof...
    prev_block = blockchain.prev_block
    proof = blockchain.proof_of_work(prev_block)

    #Receive a reward for finding the proof.
    #The sender is "0" to signify a new transaction.
    blockchain.new_trans(
        sender="0",
        recipient=node_id,
        amount=1,
    )

    #Forge the new Block by adding it to the chain
    prev_hash = blockchain.hash(prev_block)
    block = blockchain.new_block(proof, prev_hash)

    response = {
        'message': "New Block Forged",
        'index': block['index'],
        'transactions': block['transactions'],
        'proof': block['proof'],
        'prev_hash': block['prev_hash'],
    }
    return jsonify(response), 200


@app.route('/transactions/new', methods=['POST'])
def new_trans():
    values = request.get_json()

    #Check that the required fields are in the POST'ed data
    required = ['sender', 'recipient', 'amount']
    if not all(k in values for k in required):
        return 'Missing values', 400

    #Create a new Transaction
    index = blockchain.new_trans(values['sender'], values['recipient'], values['amount'])

    response = {'message': f'Transaction will be added to Block {index}'}
    
    #Kmeans clustering is implemented on the newly formed chain


    #Building the k-means model

    kmeans = KMeans(n_clusters=2)
    kmeans.fit(x_train)
    KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
    n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
    random_state=None, tol=0.0001, verbose=0)
    correct = 0
    for i in range(len(x_test)):
        predict_me = np.array(test_x[i].astype(float))
        predict_me = predict_me.reshape(-1, len(predict_me))
        prediction = kmeans.predict(predict_me)
        if prediction[0] == y[i]:
            correct += 1

        print(correct/len(x_test))
    return jsonify(response), 201

 #fit model no training data
    model = XGBClassifier()

@app.route('/chain', methods=['GET'])
def full_chain():
    response = {
        'chain': blockchain.chain,
        'length': len(blockchain.chain),
    }
    return jsonify(response), 200
    
@app.route('/nodes/register', methods=['POST'])
if requests.methods=='POST':
	print('method is post')
	def new_nodes():
	    values = request.get_json()

	    nodes = values.get('nodes')
	    if nodes is None:
	        return "Error: Please supply a valid list of nodes", 400

	    for node in nodes:
	        blockchain.new_node(node)

	    response = {
	        'message': 'New nodes have been added',
	        'total_nodes': list(blockchain.nodes),
	    }
	    return jsonify(response), 201
else:
	print('failed')


@app.route('/nodes/resolve', methods=['GET'])
def consensus():
    replaced = blockchain.conflict_resolution()

    if replaced:
        response = {
            'message': 'Our chain was replaced',
            'new_chain': blockchain.chain
        }
    else:
        response = {
            'message': 'Our chain is authoritative',
            'chain': blockchain.chain
        }

    return jsonify(response), 200


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=5000, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='localhost', port=port)