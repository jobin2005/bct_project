from Models.Block import Block

class Node(object):

    """ Defines the base Node model.

        :param int id: the uinque id of the node
        :param list blockchain: the local blockchain (a list to store chain state locally) for the node
        :param list transactionsPool: the transactions pool. Each node has its own pool if and only if Full technique is chosen
        :param int blocks: the total number of blocks mined in the main chain
        :param int balance: the amount of cryptocurrencies a node has
    """
    def __init__(self,id):
        self.id= id
        self.blockchain= []
        self.transactionsPool= []
        self.blocks= 0#
        self.balance= 0



        #telemetry data
        self.vote_delay = 0
        self.missed_votes = 0
        self.total_epochs = 0
        self.uptime = 1
        self.connectivity_degree = 0
        self.last_vote_time = 0
        self.last_block_time = 0
        self.neighbors=[]
        self.vote_delay = event_time - block.timestamp
    # Generate the Genesis block and append it to the local blockchain for all nodes
    def generate_gensis_block():
        from InputsConfig import InputsConfig as p
        for node in p.NODES:
            node.blockchain.append(Block())

    # Get the last block at the node's local blockchain
    def last_block(self):
        return self.blockchain[len(self.blockchain)-1]

    # Get the length of the blockchain (number of blocks)
    def blockchain_length(self):
        return len(self.blockchain)-1

    # reset the state of blockchains for all nodes in the network (before starting the next run) 
    def resetState():
        from InputsConfig import InputsConfig as p
        for node in p.NODES:
            node.blockchain= [] # create an array for each miner to store chain state locally
            node.transactionsPool= []
            node.blocks=0 # total number of blocks mined in the main chain
            node.balance= 0 # to count all reward that a miner made



    def update_uptime(self):
        import random

        outage_prob = 0.03
        if random.random() < outage_prob:
            self.uptime = 0
        else:
            self.uptime = 1


    def connectivity_degree(self):
        return len(self.neighbors)
    

    def missed_vote_rate(self):
        if self.total_votes == 0:
            return 0
        return self.missed_votes / self.total_votes
    
    def update_vote_delay(self):
        import random
        self.vote_delay = random.uniform(0.1, 2.0)
