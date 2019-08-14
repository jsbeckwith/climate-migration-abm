import random
import pickle
from statistics import mean
import math
from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import networkx as nx
import numpy as np
import pandas as pd


# ACCESS NODE OF AGENT BY USING .pos
# ACCESS AGENTS IN NODE BY USING G.node[n]['agent']
# ACCESS NODE DATA USING G.node[n]['attr']

TICK = 0  # counter for current time step

class ClimateMigrationModel(Model):
    def __init__(   self, num_counties, preferences=False, network_type='random', \
                    climate_threshold=[0, 75, 200], limited_radius=True, init_time=0):
        super().__init__()
        global TICK
        TICK = init_time
        self.num_agents = 0
        self.agent_index = 0
        self.preferences = preferences
        self.limited_radius = limited_radius
        self.upper_network_size = 3
        self.network_type = network_type
        self.climate_threshold = climate_threshold
        self.schedule = SimultaneousActivation(self)
        self.G = create_graph()
        self.num_counties = num_counties
        self.nodes = self.G.nodes()
        self.grid = NetworkGrid(self.G)
        self.county_climate_ranking = []
        self.county_population_list = [0] * self.num_counties
        self.county_influx = [0] * self.num_counties
        self.deaths = []
        self.births = []
        self.county_income = {}
        self.datacollector = DataCollector(model_reporters={"County Population": lambda m1: list(m1.county_population_list),
                                                            "County Influx": lambda m2: list(m2.county_influx),
                                                            "Deaths": lambda m3: m3.deaths,
                                                            "Births": lambda m4: m4.births,
                                                            "Total Population": lambda m5: m5.num_agents})

    def add_agents(self):
        """
        Adds agents based on 2013 ACS population data.
        """
        cumulative_population_list = get_cumulative_population_list()
        self.county_population_list = get_population_list()

        county = 0   # keeps track of which county each agent should be placed in
        index = 0   # keeps track of each agent's unique_id

        # keep creating agents until county population is reached
        while index < cumulative_population_list[county]:
            # create agent
            agent = Household(index, self)
            # place agent in appropriate county
            self.grid.place_agent(agent, list(self.nodes)[county])
            # add agent to schedule
            self.schedule.add(agent)
            # set agent's original_pos attribute
            agent.original_pos = agent.pos
            # initialize all other agent attributes
            agent.initialize_agent()
            # if running model with heterogeneous preferences, set agent preference
            if self.preferences:
                agent.initialize_preference()

            # update index
            index += 1

            # if done with county and not at last county, increase county
            if index == cumulative_population_list[county] and county < self.num_counties - 1:
                county += 1

        # after all agents are added, set model attributes
        self.num_agents = cumulative_population_list[self.num_counties-1]
        self.agent_index = cumulative_population_list[self.num_counties-1]

    def initialize_all_networks(self):
        """
        Initializes random networks for all agents in model.
        """
        for a in self.schedule.agents:
            a.initialize_network()

    def initialize_all_income_networks(self):
        """
        Initializes income-based networks for all agents in model.
        """
        for a in self.schedule.agents:
            a.initialize_income_network()

    def initialize_all_age_networks(self):
        """
        Initializes age-based networks for all agents in model.
        """
        for a in self.schedule.agents:
            a.initialize_age_network()

    def initialize_all_income_age_networks(self):
        """
        Initializes income and age-based networks for all agents in model.
        """
        for a in self.schedule.agents:
            a.initialize_income_age_network()

    def initialize_all_families(self):
        """
        Initializes families for all agents in model.
        """
        for a in self.schedule.agents:
            a.initialize_family()

    def update_population(self):
        """
        Updates population by adding and removing agents.
        """
        # keep track of number of deaths and births per county
        self.deaths = [0]*self.num_counties
        self.births = [0]*self.num_counties

        # remove agents (death)
        # loop through all agents
        for agent in self.schedule.agents:
            # source: https://www.ssa.gov/oact/STATS/table4c6.html#ss
            # calculate death probability by age in the united states
            if random.random() < 0.0001*(math.e**(0.075*agent.age)):
                # keep track of deaths by county
                self.deaths[agent.pos] += 1
                # remove agent from model
                self.grid._remove_agent(agent, agent.pos)
                # remove agent from schedule
                self.schedule.remove(agent)
                # update number of agents
                self.num_agents -= 1

        # add agents (birth)
        # loop through all counties
        for county in range(self.num_counties):
            # source: https://www.cdc.gov/nchs/fastats/births.htm
            # access current population
            current_population = self.county_population_list[county]
            # calculate how many agents should be added
            to_add = current_population//100 # birth rate
            # update number of agents
            self.num_agents += to_add

            # add specified number of agents
            for i in range(to_add):
                # update agent index
                self.agent_index += 1
                # create new agent
                agent = Household(self.agent_index, self)
                # place agent in current county
                self.grid.place_agent(agent, county)
                # add agent to schedule
                self.schedule.add(agent)

                # initialize agent attributes and networks
                
                # agents are assumed to be 18 as that is the lower bound of a householder's age
                agent.age = 18
                # based on age, income is assigned
                agent.initialize_income(random.random())
                # based on income, tenure is assigned
                agent.initialize_tenure(random.random())
                # input-specified network is initialized
                agent.initialize_network()
                # family is initialized
                agent.initialize_family()
                # original position is set
                agent.original_pos = agent.pos
                # keep track of births by county
                self.births[agent.pos] += 1

        # loop through nodes, update population counts
        for node in self.nodes:
            self.county_population_list[node] = len(self.G.node[node]['agent'])

    def update_climate(self):
        """
        Update climate variables based on NOAA's predictions.

        Index 1 represents number of days above 90 degrees Fahrenheit.
        Index 4 represents number of days with <1 inches of rain.
        Index 7 represents number of days without rain.
        Indexes 3, 6, and 9 are the yearly increases/decreases for these estimates.
        The update function is a simple linear function.
        
        More accurate climate data could be integrated by importing more
        climate explorer data.
        """
        for n in self.nodes:
            self.G.node[n]['climate'][1] += self.G.node[n]['climate'][3]
            self.G.node[n]['climate'][4] += self.G.node[n]['climate'][6]
            self.G.node[n]['climate'][7] += self.G.node[n]['climate'][9]

    def rank_by_climate(self):
        # necessary ? - how to collect/show model-level data is the bigger question
        heat_data = []
        dry_data = []
        heat_dry_data = []

        for n in self.nodes:
            heat_data.append(self.G.node[n]['climate'][1])
            dry_data.append(self.G.node[n]['climate'][7])

        max_heat = max(heat_data)
        max_dry = max(dry_data)
        heat_data = [(e/max_heat) for e in heat_data]
        dry_data = [(e/max_dry) for e in dry_data]

        for i in range(self.num_counties):
            heat_dry_data.append(heat_data[i] + dry_data[i])

        heat_dry_data = np.array(heat_dry_data)
        county_climate_rank = np.argsort(heat_dry_data)
        self.county_climate_ranking = list(county_climate_rank)

    def update_income_counts(self):
        for i in range(self.num_counties):
            self.county_income[i] = [0]*10
        for a in self.schedule.agents:
            self.county_income[a.pos][a.income-1] += 1
        print(self.county_income)

    def step(self):
        global TICK
        self.rank_by_climate()
        self.schedule.step()
        self.update_population()
        self.update_climate()
        self.datacollector.collect(self)
        TICK += 1


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.original_pos = None
        self.age = 0
        self.probability = [0] * 3 # age, tenure, children
        self.income = 0
        self.tenure = 0 # 0 = own, 1 = rent
        self.preference = 0 # 1 = climate, 2 = family, 3 = network, 4 = cost of living
        self.connections = []  # list of all connected agents
        self.family = []
        self.counties_by_network = []
        self.adaptive_capacity = 0  # ability to implement adaptation actions

    def initialize_agent(self):
        self.initialize_age(random.random())
        self.initialize_income(random.random())
        self.initialize_tenure(random.random())

    def initialize_age(self, rand_num):
        age_list = self.model.G.node[self.pos]['age']
        if rand_num < age_list[0]:
            self.age = random.randint(18, 24)
        elif rand_num < age_list[1]:
            self.age = random.randint(25, 44)
        elif rand_num < age_list[2]:
            self.age = random.randint(45, 64)
        else:
            self.age = random.randint(65, 80)

    def initialize_income(self, rand_num):
        if self.age < 25:
            income_list = self.model.G.node[self.pos]['u25income']
        elif self.age < 45:
            income_list = self.model.G.node[self.pos]['income2544']
        elif self.age < 65:
            income_list = self.model.G.node[self.pos]['income4564']
        else:
            income_list = self.model.G.node[self.pos]['income65a']

        if rand_num < income_list[0]:
            self.income = 1
        elif rand_num < income_list[1]:
            self.income = 2
        elif rand_num < income_list[2]:
            self.income = 3
        elif rand_num < income_list[3]:
            self.income = 4
        elif rand_num < income_list[4]:
            self.income = 5
        elif rand_num < income_list[5]:
            self.income = 6
        elif rand_num < income_list[6]:
            self.income = 7
        elif rand_num < income_list[7]:
            self.income = 8
        elif rand_num < income_list[8]:
            self.income = 9
        else:
            self.income = 10

    def initialize_tenure(self, rand_num):
        tenure_list = self.model.G.node[self.pos]['tenure']
        if rand_num > tenure_list[self.income-1]:
            self.tenure = 1

    def initialize_preference(self):
        self.preference = random.randint(1, 5)

    def initialize_network(self):
        if self.model.network_type == 'income':
            self.initialize_income_network()
        elif self.model.network_type == 'age':
            self.initialize_age_network()
        elif self.model.network_type == 'income_age':
            self.initialize_income_age_network()
        else:
            upper_bound = self.model.upper_network_size
            self.connections = random.sample(self.model.G.node[self.pos]['agent'], upper_bound)
            self.connections += random.sample(self.model.schedule.agents, random.randint(1, upper_bound))

    def initialize_income_network(self):
        upper_bound = self.model.upper_network_size
        while len(self.connections) < upper_bound:
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            if abs(potential_connection.income - self.income) <= 1:
                self.connections.append(potential_connection)
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            potential_connection = random.choice(self.model.schedule.agents)
            if abs(potential_connection.income - self.income) <= 1:
                self.connections.append(potential_connection)

    def initialize_age_network(self):
        upper_bound = self.model.upper_network_size
        while len(self.connections) < upper_bound:
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            if abs(potential_connection.age - self.age) <= 5:
                self.connections.append(potential_connection)
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            potential_connection = random.choice(self.model.schedule.agents)
            if abs(potential_connection.age - self.age) <= 5:
                self.connections.append(potential_connection)

    def initialize_income_age_network(self):
        upper_bound = self.model.upper_network_size
        while len(self.connections) < upper_bound:
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            if abs(potential_connection.age - self.age) <= 5:
                if abs(potential_connection.income - self.income) <= 1:
                    self.connections.append(potential_connection)
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            potential_connection = random.choice(self.model.schedule.agents)
            if abs(potential_connection.age - self.age) <= 5:
                if abs(potential_connection.income - self.income) <= 1:
                    self.connections.append(potential_connection)

    def initialize_family(self):
        while not self.family:
            if self.income < 8:
                potential_family = random.choice(self.model.G.node[self.pos]['agent'])
                differential = random.randint(17, 23)
            else:
                potential_family = random.choice(self.model.schedule.agents)
                differential = random.randint(20, 26)
            if abs(self.age - potential_family.age) > differential:
                self.family.append(potential_family)
                potential_family.family.append(self)

    def update_age(self):
        self.age += 1

    def update_income_tenure(self):
        if self.age == 26:
            if self.income < 3:
                self.income += random.randint(-1, 6)
            if self.income * 0.0715 > random.random():
                self.tenure = 0
        elif self.age == 46:
            if self.income == 7:
                self.income += random.randint(-1, 2)
                if self.income * 0.0715 > random.random():
                    self.tenure = 0
        elif self.age == 66:
            if self.income > 7:
                self.income -= random.randint(0, 2)

    def update_network(self):
        if self.model.network_type == 'income':
            self.update_income_network()
        elif self.model.network_type == 'age':
            self.update_age_network()
        elif self.model.network_type == 'income_age':
            self.update_income_age_network()
        else:
            if len(self.connections) < self.model.upper_network_size*2:
                if random.random() < 0.5:
                    self.connections.append(random.choice(self.model.G.node[self.pos]['agent']))

    def update_income_network(self):
        current_network_size = len(self.connections)
        if current_network_size < self.model.upper_network_size*2:
            if random.random() < 0.5:
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.income - self.income) <= 1:
                        self.connections.append(potential_connection)

    def update_age_network(self):
        current_network_size = len(self.connections)
        if current_network_size < self.model.upper_network_size*2:
            if random.random() < 0.5:
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.age - self.age) <= 5:
                        self.connections.append(potential_connection)

    def update_age_income_network(self):
        current_network_size = len(self.connections)
        if current_network_size < self.model.upper_network_size*2:
            if random.random() < 0.5:
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.age - self.age) <= 5:
                        if abs(potential_connection.income - self.income) <= 1:
                            self.connections.append(potential_connection)

    def rank_counties_by_network(self):
        connected_locations = []
        self.counties_by_network = [0]*self.model.num_counties
        for i in range(len(self.connections)):
            connected_locations.append(self.connections[i].pos)
        for n in connected_locations:
            self.counties_by_network[n] += 1

    def calculate_migration_probability(self):
        # data from census report 2005-2010
        if self.age < 25:
            self.probability[0] += 0.179
        elif self.age < 29:
            self.probability[0] += 0.248
        elif self.age < 45:
            self.probability[0] += 0.156
        elif self.age < 65:
            self.probability[0] += 0.082
        elif self.age < 75:
            self.probability[0] += 0.064
        else:
            self.probability[0] += 0.046

        if self.tenure == 0:
            self.probability[1] += 0.084
        else:
            self.probability[1] += 0.212

        self.probability[2] += random.uniform(-0.1, 0.1)

    def calculate_adaptive_capacity(self):
        self.adaptive_capacity = self.income/10
        if self.age > 65:
            self.adaptive_capacity -= self.age/600

    def get_counties_in_radius(self, to_choose):
        radius = (self.adaptive_capacity) * 3000

        for i in range(self.model.num_counties):
            if i != self.pos:
                distance = self.model.G.get_edge_data(self.pos, i)['distance']
                if distance < radius:
                    for j in range(3000//distance):
                        to_choose.append(i)

        return to_choose

    def get_all_counties(self, to_choose):
        for i in range(self.model.num_counties):
            to_choose.append(i)
        return to_choose

    def get_family_counties(self, to_choose):
        to_choose.append(self.family[0].pos)

        if self.preference == 2:
            for k in range(3):
                to_choose.append(self.family[0].pos)

        return to_choose

    def get_network_counties(self, to_choose):
        for i in range(len(self.counties_by_network)):

            for j in range(self.counties_by_network[i]):
                to_choose.append(i)

            if self.preference == 3:
                for k in range(3):
                    to_choose.append(i)

        return to_choose

    def get_counties_by_price(self, to_choose):
        for county in to_choose:
            median_house = self.model.G.node[county]['climate'][14]
            print(median_house)
            if self.income < 7:
                if median_house < self.income:
                    to_choose.append(county)
                if self.preference == 4:
                    for k in range(self.income - median_house):
                        to_choose.append(county)

            else:
                if self.income - median_house < 2:
                    to_choose.append(county)
                if self.preference == 4:
                    if median_house < self.income:
                        for k in range(self.income - median_house):
                            to_choose.append(county)

        return to_choose

    def add_climate_counties(self, current_county_climate_rank, to_choose):
        for i in range(current_county_climate_rank):
            county = self.model.county_climate_ranking[i]
            if county in to_choose:
                to_choose.append(county)
                for j in range(3//(i+1)): # weighted component
                    to_choose.append(county)

        for i in range(current_county_climate_rank, self.model.num_counties):
            county = self.model.county_climate_ranking[i]
            if county in to_choose:
                to_choose.remove(county)

        return to_choose

    def remove_sea_level_counties(self, to_choose):
        for county in to_choose:
            if self.model.G.node[county]['climate'][11] > 0:
                if random.random() < self.model.G.node[county]['climate'][11]:
                    to_choose.remove(county)

        return to_choose

    def make_climate_decision(self):
        if random.random() < mean(self.probability):
            to_choose = []
            to_move = None

            if self.model.limited_radius:
                to_choose = self.get_counties_in_radius(to_choose)
            else:
                to_choose = self.get_all_counties(to_choose)

            to_choose = self.get_counties_by_price(to_choose)
            to_choose = self.get_family_counties(to_choose)

            self.rank_counties_by_network()

            to_choose = self.get_network_counties(to_choose)

            heat = self.model.G.node[self.pos]['climate'][1]
            dry = self.model.G.node[self.pos]['climate'][7]
            current_county_climate_rank = self.model.county_climate_ranking.index(self.pos)

            if self.model.climate_threshold[0] == 0:
                heat_threshold = self.model.climate_threshold[1]
                dry_threshold = self.model.climate_threshold[2]
                if heat > heat_threshold and dry > dry_threshold or self.preference == 1:
                    to_choose = self.add_climate_counties(current_county_climate_rank, to_choose)

            elif self.model.climate_threshold[0] == 1:
                relative_threshold = self.model.climate_threshold[1]
                if current_county_climate_rank > relative_threshold or self.preference == 1:
                    to_choose = self.add_climate_counties(current_county_climate_rank, to_choose)

            to_choose = self.remove_sea_level_counties(to_choose)

            if to_choose:
                to_move = random.choice(to_choose)
                if to_move != self.pos:
                    self.model.county_influx[self.pos] -= 1
                    self.model.county_influx[to_move] += 1
                    # track migration between counties
                    if self.pos > to_move:
                        self.model.G[self.pos][to_move]['net_mig'] += 1
                    else:
                        self.model.G[to_move][self.pos]['net_mig'] -= 1
                    self.model.grid.move_agent(self, to_move)

    def step(self):
        self.update_age()
        self.update_income_tenure()
        self.update_network()
        self.calculate_migration_probability()

    def advance(self):
        self.make_climate_decision()

def create_graph():
    with open('pickle/real_data_dict_slr_house.pickle', 'rb') as node_data_file:
        node_data = pickle.load(node_data_file)

    with open('pickle/distance_dict.pickle', 'rb') as edge_data_file:
        edge_data = pickle.load(edge_data_file)

    with open('pickle/migration_pair_dict.pickle', 'rb') as mig_data_file:
        migration_edge_data = pickle.load(mig_data_file)

    G = nx.complete_graph(74)
    nx.set_node_attributes(G, node_data)
    nx.set_edge_attributes(G, edge_data, 'distance')
    nx.set_edge_attributes(G, migration_edge_data, 'net_mig')

    return G

def get_population_list():
    population_data = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    population_dictionary = population_data.to_dict('list')
    return population_dictionary['total_pop_1000']

def get_cumulative_population_list():
    population_data = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    cumulative_population = population_data.cumsum(axis=0, skipna=True)
    cumulative_population_dictionary = cumulative_population.to_dict('list')
    return cumulative_population_dictionary['total_pop_1000']
