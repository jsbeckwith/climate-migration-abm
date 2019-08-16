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
        self.county_flux = [0] * self.num_counties
        self.deaths = []
        self.births = []
        self.county_income = {}
        self.datacollector = DataCollector(model_reporters={"County Population": lambda m1: list(m1.county_population_list),
                                                            "County Influx": lambda m2: list(m2.county_flux),
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
            for count in range(to_add):
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

        # loop through counties, update population counts
        for county in self.nodes:
            self.county_population_list[county] = len(self.G.node[county]['agent'])

    def update_climate(self):
        """
        Update climate variables based on NOAA's predictions.

        Index 1 represents number of days above 90 degrees Fahrenheit.
        Index 4 represents number of days with < 1 inch of rain.
        Index 7 represents number of days without rain.
        Indexes 3, 6, and 9 are the yearly increases/decreases for these estimates.
        The update function is a simple linear function.
        
        Note: More accurate climate data could be integrated by importing
        more climate explorer data.
        """
        for n in self.nodes:
            self.G.node[n]['climate'][1] += self.G.node[n]['climate'][3]
            self.G.node[n]['climate'][4] += self.G.node[n]['climate'][6]
            self.G.node[n]['climate'][7] += self.G.node[n]['climate'][9]

    def rank_by_climate(self):
        """
        Create an ordered list of counties, from least hot/dry climate to
        most hot/dry climate.
        """
        # initialize lists to store data
        heat_data = []
        dry_data = []
        heat_dry_data = []

        # loop through counties in order
        for county in self.nodes:
            # access and store all heat/dry data
            heat_data.append(self.G.node[county]['climate'][1])
            dry_data.append(self.G.node[county]['climate'][7])

        # find max heat/dry data
        max_heat = max(heat_data)
        max_dry = max(dry_data)

        # normalize data based on max value
        heat_data = [(e/max_heat) for e in heat_data]
        dry_data = [(e/max_dry) for e in dry_data]

        # add normalized data
        for county in range(self.num_counties):
            heat_dry_data.append(heat_data[county] + dry_data[county])

        # convert to numpy array
        heat_dry_data = np.array(heat_dry_data)
        # returns indices that would sort an array (in this case,
        # returns county id's from best to worst climate)
        county_climate_rank = np.argsort(heat_dry_data)
        # convert to list, update model attribute
        self.county_climate_ranking = list(county_climate_rank)

    def update_income_counts(self):
        """
        Update income distribution by county.
        """
        # loop through counties in order
        for county in range(self.num_counties):
            # initialize list
            self.county_income[county] = [0]*10
        # loop through agents
        for agent in self.schedule.agents:
            # update dictionary based on agent data
            self.county_income[agent.pos][agent.income-1] += 1
        # income counts are printed at the beginning and end of run
        print(self.county_income)

    def step(self):
        """
        Advance the model by one step.
        """
        global TICK
        # update climate ranking
        self.rank_by_climate()
        # advance all agents by one step
        self.schedule.step()
        # update population
        self.update_population()
        # update climate
        self.update_climate()
        # collect data
        self.datacollector.collect(self)
        # update step counter
        TICK += 1


class Household(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.original_pos = None
        self.age = 0
        self.probability = 0
        # 1 = <10k, 2 = 10k-15k, 3 = 15k-20k, 4 = 20k-25k, 5 = 25k-25k,
        # 6 = 35k-50k, 7 = 50k-75k, 8 = 75k-100k, 9 = 100k-150k, 10 = >150k
        self.income = 0
        self.tenure = 0 # 0 = own, 1 = rent
        self.preference = 0 # 1 = climate, 2 = family, 3 = network, 4 = cost of living
        self.connections = []  # list of all connected agents
        self.family = []
        self.connected_locations = []


    ####################################################################################
    #                           INITIALIZATION FUNCTIONS                               #
    ####################################################################################

    def initialize_agent(self):
        """
        Initialize all non-network attributes of an agent.
        Age informs income, which then informs tenure.
        """
        self.initialize_age(random.random())
        self.initialize_income(random.random())
        self.initialize_tenure(random.random())

    def initialize_age(self, rand_num):
        """
        Initialize an agent's age based on 2013 ACS data.
        """
        # access age distribution for the agent's county
        age_list = self.model.G.node[self.pos]['age']
        # probability of being 18-24
        if rand_num < age_list[0]:
            self.age = random.randint(18, 24)
        # probability of being 25-44
        elif rand_num < age_list[1]:
            self.age = random.randint(25, 44)
        # probability of being 45-64
        elif rand_num < age_list[2]:
            self.age = random.randint(45, 64)
        # probability of being 65-80
        else:
            self.age = random.randint(65, 80)

    def initialize_income(self, rand_num):
        """
        Initialize an agent's income based on 2013 ACS data.
        """
        # access correct dataset based on agent's age and county
        if self.age < 25:
            income_list = self.model.G.node[self.pos]['u25income']
        elif self.age < 45:
            income_list = self.model.G.node[self.pos]['income2544']
        elif self.age < 65:
            income_list = self.model.G.node[self.pos]['income4564']
        else:
            income_list = self.model.G.node[self.pos]['income65a']

        # probability of income less than 10k
        if rand_num < income_list[0]:
            self.income = 1
        # probability of income 10k-15k
        elif rand_num < income_list[1]:
            self.income = 2
        # probability of income 15k-20k
        elif rand_num < income_list[2]:
            self.income = 3
        # probability of income 20k-25k
        elif rand_num < income_list[3]:
            self.income = 4
        # probability of income 25k-35k
        elif rand_num < income_list[4]:
            self.income = 5
        # probability of income 35k-50k
        elif rand_num < income_list[5]:
            self.income = 6
        # probability of income 50k-75k
        elif rand_num < income_list[6]:
            self.income = 7
        # probability of income 75k-100k
        elif rand_num < income_list[7]:
            self.income = 8
        # probability of income 100k-150k
        elif rand_num < income_list[8]:
            self.income = 9
        # probability of income >150k
        else:
            self.income = 10

    def initialize_tenure(self, rand_num):
        """
        Initialize an agent's tenure based on 2013 ACS data.
        """
        # access correct dataset based on agent's county
        tenure_list = self.model.G.node[self.pos]['tenure']
        # probability of owning a house based on income
        if rand_num > tenure_list[self.income-1]:
            self.tenure = 1

    def initialize_preference(self):
        # TODO: preference distribution, docstring, comments
        self.preference = random.randint(1, 5)

    def initialize_network(self):
        """
        Initialize agent's network, according to type specified upon model creation.
        """
        if self.model.network_type == 'income':
            self.initialize_income_network()
        elif self.model.network_type == 'age':
            self.initialize_age_network()
        elif self.model.network_type == 'income_age':
            self.initialize_income_age_network()
        # initialize random network
        else:
            self.initialize_random_network()

    def initialize_random_network(self):
        """
        Initialize random network.
        """
        # get maximum network size
        upper_bound = self.model.upper_network_size
        # while network size is below maximum, find agents in same county
        while len(self.connections) < upper_bound:
            # choose a random agent from the same county
            connection = random.choice(self.model.G.node[self.pos]['agent'])
            # connect agents
            self.connections.append(connection)
            connection.connections.append(self)
        # while network size is below maximum, find agents in any county
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            # choose a random agent from all counties
            connection = random.choice(self.model.schedule.agents)
            # connect agents
            self.connections.append(connection)
            connection.connections.append(self)

    def initialize_income_network(self):
        """
        Initialize income-based network.
        """
        # get maximum network size
        upper_bound = self.model.upper_network_size
        # while network size is below maximum, find agents with similar income in same county
        while len(self.connections) < upper_bound:
            # choose a random agent from the same county
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            # if neighboring income, add to network & vice-versa
            if abs(potential_connection.income - self.income) <= 1:
                self.connections.append(potential_connection)
                potential_connection.connections.append(self)
        # while network size is below maximum, find agents with similar income in any county
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            # choose a random agent from all counties
            potential_connection = random.choice(self.model.schedule.agents)
            # if neighboring income, add to network & vice-versa
            if abs(potential_connection.income - self.income) <= 1:
                self.connections.append(potential_connection)
                potential_connection.connections.append(self)

    def initialize_age_network(self):
        """
        Initialize age-based network
        """
        # get maximum network size
        upper_bound = self.model.upper_network_size
        # while network size is below maximum, find agents with similar age in same county
        while len(self.connections) < upper_bound:
            # choose a random agent from the same county
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            # if close in age, add to network & vice-versa
            if abs(potential_connection.age - self.age) <= 5:
                self.connections.append(potential_connection)
                potential_connection.connections.append(self)
        # while network size is below maximum, find agents with similar age in any county
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            # choose a random agent from all counties
            potential_connection = random.choice(self.model.schedule.agents)
            # if close in age, add to network & vice-versa
            if abs(potential_connection.age - self.age) <= 5:
                self.connections.append(potential_connection)
                potential_connection.connections.append(self)

    def initialize_income_age_network(self):
        """
        Initialize income and age-based network.
        """
        # get maximum network size
        upper_bound = self.model.upper_network_size
        # while network size is below maximum, find agents with similar income/age in same county
        while len(self.connections) < upper_bound:
            # choose a random agent from the same county
            potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
            # if close in income/age, add to network & vice-versa
            if abs(potential_connection.age - self.age) <= 5:
                if abs(potential_connection.income - self.income) <= 1:
                    self.connections.append(potential_connection)
                    potential_connection.connections.append(self)
        # while network size is below maximum, find agents with similar age in any county
        while len(self.connections) < upper_bound + random.randint(1, upper_bound):
            # choose a random agent from all counties
            potential_connection = random.choice(self.model.schedule.agents)
            # if close in income/age, add to network & vice-versa
            if abs(potential_connection.age - self.age) <= 5:
                if abs(potential_connection.income - self.income) <= 1:
                    self.connections.append(potential_connection)
                    potential_connection.connections.append(self)

    def initialize_family(self):
        """
        Initialize family.
        """
        # while no family connections, find potential family members
        while not self.family:
            # if lower-income, choose a random agent from the same county
            if self.income < 8:
                potential_family = random.choice(self.model.G.node[self.pos]['agent'])
                # randomly choose age difference lower bound
                differential = random.randint(17, 23)
            # if higher-income, choose a random agent from any county
            else:
                potential_family = random.choice(self.model.schedule.agents)
                # randomly choose age difference lower bound
                differential = random.randint(20, 26)
            # if age difference checks out, update both agents' families
            if abs(self.age - potential_family.age) > differential:
                self.family.append(potential_family)
                potential_family.family.append(self)


    ####################################################################################
    #                                 UPDATE FUNCTIONS                                 #
    ####################################################################################

    def update_age(self):
        """
        Update age.
        """
        self.age += 1

    def update_income_tenure(self):
        """
        Update income and tenure as agents move into new age groups.
        """
        if self.age == 26:
            # randomly update income, with a skew towards upward mobility
            if self.income < 3:
                self.income += random.randint(-1, 5)
            # linear relationship between income and tenure, according to ACS data
            if self.income * 0.0715 > random.random():
                self.tenure = 0
        elif self.age == 46:
            # randomly update income, with a slight skew towards upward mobility
            if self.income == 7:
                self.income += random.randint(-1, 2)
                # if income updated, update tenure as well
                if self.income * 0.0715 > random.random():
                    self.tenure = 0
        elif self.age == 66:
            # randomly update income, with a skew towards downward mobility
            if self.income > 7:
                self.income -= random.randint(0, 2)

    def update_network(self):
        """
        Update agent's networks, according to type specified upon model creation. 
        """
        if self.model.network_type == 'income':
            self.update_income_network()
        elif self.model.network_type == 'age':
            self.update_age_network()
        elif self.model.network_type == 'income_age':
            self.update_income_age_network()
        else:
            self.update_random_network()
            

    def update_random_network(self):
        """
        Update agent's random network.
        """
        # if network isn't too big, there is a 30% chance of adding a new agent
        if len(self.connections) < self.model.upper_network_size*4:
            if random.random() < 0.3:
                # randomly choose agent in same county
                connection = random.choice(self.model.G.node[self.pos]['agent'])
                # connect agents
                self.connections.append(connection)
                connection.connections.append(self)

    def update_income_network(self):
        """
        Update agent's income-based network.
        """
        current_network_size = len(self.connections)
        # if network isn't too big, there is a 30% chance of adding a new agent
        if current_network_size < self.model.upper_network_size*4:
            if random.random() < 0.3:
                # iterate through randomly-chosen agents in same county until 
                # agent with similar income is found and added
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.income - self.income) <= 1:
                        # connect agents
                        self.connections.append(potential_connection)
                        potential_connection.connections.append(self)

    def update_age_network(self):
        """
        Update agent's age-based network.
        """
        current_network_size = len(self.connections)
        # if network isn't too big, there is a 30% chance of adding a new agent
        if current_network_size < self.model.upper_network_size*4:
            if random.random() < 0.3:
                # iterate through randomly chosen agents in same county
                # until agent with similar age is found and added
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.age - self.age) <= 5:
                        # connect agents
                        self.connections.append(potential_connection)
                        potential_connection.connections.append(self)

    def update_age_income_network(self):
        """
        Update agent's income and age-based network.
        """
        current_network_size = len(self.connections)
        # if network isn't too big, there is a 30% chance of adding a new agent
        if current_network_size < self.model.upper_network_size*4:
            if random.random() < 0.3:
                # iterate through randomly chosen agents in same county
                # until agent with similar income/age is found and added
                while len(self.connections) < current_network_size+1:
                    potential_connection = random.choice(self.model.G.node[self.pos]['agent'])
                    if abs(potential_connection.age - self.age) <= 5:
                        if abs(potential_connection.income - self.income) <= 1:
                            # connect agents
                            self.connections.append(potential_connection)
                            potential_connection.connections.append(self)


    ####################################################################################
    #                          MIGRATION DECISION FUNCTIONS                            #
    ####################################################################################

    def calculate_migration_probability(self):
        """
        Calculate an agent's migration probability, based on
        their attributes. 

        Probabilities as related to age and tenure come from
        Ihkre and Faber (2012)

        Probability based on age is index 0, probability based
        on tenure is index 1, randomness factor is index 2.

        Overall probability is calculated as the average of
        these three numbers.
        """
        # initialize probability list
        probability_list = [0]*3

        if self.age < 25:
            probability_list[0] += 0.179
        elif self.age < 29:
            probability_list[0] += 0.248
        elif self.age < 45:
            probability_list[0] += 0.156
        elif self.age < 65:
            probability_list[0] += 0.082
        elif self.age < 75:
            probability_list[0] += 0.064
        else:
            probability_list[0] += 0.046

        if self.tenure == 0:
            probability_list[1] += 0.084
        else:
            probability_list[1] += 0.212

        probability_list[2] += random.uniform(-0.1, 0.1)

        # calculate and assign probability to agent
        self.probability = mean(probability_list)

    def get_counties_in_radius(self, to_choose):
        """
        Add all counties within income-dependent radius to
        list of possible counties to migrate to.
        """
        # calculate radius; 300 as a scaling factor ensures that 
        # highest-income agents will be able to migrate to any county
        radius = self.income * 300

        # loop through all counties
        for county in range(self.model.num_counties):
            # check that county is not agent's current county
            if county != self.pos:
                # get distance
                distance = self.model.G.get_edge_data(self.pos, county)['distance']
                if distance < radius:
                    # scaling factor is 2697 to ensure that furthest counties will be added once
                    weight = 2697//distance
                    # if counties are closer than 180 miles, weight defaults to 15
                    if weight > 15:
                        weight = 15
                    # add counties to list, weighted by distance
                    for count in range(weight):
                        to_choose.append(county)

        return to_choose

    def get_all_counties(self, to_choose):
        """
        Add all counties to list of possible counties
        to migrate to.
        """
        for county in range(self.model.num_counties):
            to_choose.append(county)
        return to_choose

    def get_family_counties(self, to_choose):
        """
        Add all counties with family members to list of
        possible counties to migrate to.
        """
        # all agents have one family household
        to_choose.append(self.family[0].pos)

        # if agent's preference is to stay near their family,
        # weight list appropriately
        if self.preference == 2:
            for count in range(5):
                to_choose.append(self.family[0].pos)

        return to_choose

    def get_network_counties(self, to_choose):
        """
        Add all counties with networked agents to list of
        possible counties to migrate to.
        """
        # initialize list of connected locations
        self.connected_locations = []
        
        # loop through all connected agents and add their location
        # (note that this is implicitly weighted if there are more
        # than one networked agents in the same county)
        for connection in self.connections:
            self.connected_locations.append(connection.pos)
        
        # add networked locations to to_choose
        to_choose += self.connected_locations

        # if network preference, add networked locations again
        if self.preference == 3:
            to_choose += self.connected_locations

        return to_choose

    def get_counties_by_price(self, to_choose):
        """
        Re-add counties to list of possible counties to migrate
        to based on median house price.
        """
        # loop through all counties in list
        for county in to_choose:
            # access county's median house price
            median_house = self.model.G.node[county]['climate'][14]

            # lower-income agents will prioritize lower cost of living more
            if self.income < 7:
                # re-add county if median house price is less than agent's income
                if median_house < self.income:
                    to_choose.append(county)
                # if preference is low cost of living, re-add county,
                # weighted by difference between income and house price 
                if self.preference == 4:
                    for count in range(self.income - median_house):
                        to_choose.append(county)

            # higher-income agents will not prioritize lower cost of living as much
            else:
                # only will re-add if within 2 levels of income
                if self.income - median_house < 3:
                    to_choose.append(county)
                # unless preference is low cost of living; then counties are
                # re-added and weighted by difference between income and house price
                if self.preference == 4:
                    if median_house < self.income:
                        for count in range(self.income - median_house):
                            to_choose.append(county)

        return to_choose

    def climate_filter(self, current_county_climate_rank, to_choose):
        """
        Re-add and remove counties to list of possible counties to migrate
        to based on climate data.
        """
        # loop through all counties with better climate than 
        # agent's current climate
        for index in range(current_county_climate_rank):
            # get county at current index
            county = self.model.county_climate_ranking[index]
            # if agent doesn't care about climate, don't weight as heavily
            # and only re-add counties that are already in the list
            if self.preference != 1:
                if county in to_choose:
                    to_choose.append(county)
                    # weight top three counties
                    for count in range(3//(index+1)): 
                        to_choose.append(county)
            # if agent does care about climate, weight top 7 counties
            # and add counties that weren't previously in list
            elif self.preference == 1:
                for count in range(7//(index+1)):
                    to_choose.append(county)

        # loop through all counties with worse climate than
        # agent's current climate
        for index in range(current_county_climate_rank, self.model.num_counties):
            # get county at current index
            county = self.model.county_climate_ranking[index]
            if self.preference != 1:
                if county in to_choose:
                    # remove county with worse climate ranking
                    # (note that remove will only remove one instance
                    # if county appears multiple times)
                    to_choose.remove(county)
            # if agent cares about climate, remove all instances of
            # counties with worse climate
            elif self.preference == 1:
                while county in to_choose:
                    to_choose.remove(county)

        return to_choose

    def remove_sea_level_counties(self, to_choose):
        """
        Remove counties from list of possible counties to migrate
        to based on sea-level rise data. (Hauer et. al, 2017)
        """
        # loop through all possible counties
        for county in to_choose:
            # check if county is vulnerable to sea-level rise
            if self.model.G.node[county]['climate'][11] > 0:
                # if agent doesn't care about climate
                # remove all instances of county if random number is less than
                # percent of county that will be affected by sea level rise
                if self.preference != 1:
                    if random.random() < self.model.G.node[county]['climate'][11]:
                        while county in to_choose:
                            to_choose.remove(county)
                # if agent cares about climate, remove all instances of county
                elif self.preference == 1:
                    while county in to_choose:
                        to_choose.remove(county)

        return to_choose

    def make_migration_decision(self):
        """
        Migration decision function.

        High-level overview: creates a list of possible counties to migrate 
        to based on cost of living, networks, family, income-based radius, 
        climate, and weighs list according to agent's preferences. 
        Randomly selects a county from the list for the agent to migrate to.
        """
        # if random number is less than migration probability, run function
        if random.random() < self.probability:
            # initialize list and county to move to
            to_choose = []
            to_move = None

            # if limited radius parameter is True, get counties in radius
            if self.model.limited_radius:
                to_choose = self.get_counties_in_radius(to_choose)
            # else get all counties and add to to_choose
            else:
                to_choose = self.get_all_counties(to_choose)

            # update to_choose based on median house price, family, and network
            to_choose = self.get_counties_by_price(to_choose)
            to_choose = self.get_family_counties(to_choose)
            to_choose = self.get_network_counties(to_choose)

            # access climate data for the agent's current county: heat days, dry
            # days, and county's current relative climate rank
            heat = self.model.G.node[self.pos]['climate'][1]
            dry = self.model.G.node[self.pos]['climate'][7]
            current_county_climate_rank = self.model.county_climate_ranking.index(self.pos)

            # if absolute threshold, all agents in counties above user-specified
            # climate threshold and agents with a climate preference will have their
            # list of counties subjected to climate review
            if self.model.climate_threshold[0] == 0:
                # user-specified threshold
                heat_threshold = self.model.climate_threshold[1]
                dry_threshold = self.model.climate_threshold[2]
                if heat > heat_threshold and dry > dry_threshold or self.preference == 1:
                    to_choose = self.climate_filter(current_county_climate_rank, to_choose)

            # if relative threshold, all agents in counties above user-specified
            # climate index and agents with a climate preference will have their
            # list of counties subjected to climate review
            elif self.model.climate_threshold[0] == 1:
                # user-specified threshold
                relative_threshold = self.model.climate_threshold[1]
                if current_county_climate_rank > relative_threshold or self.preference == 1:
                    to_choose = self.climate_filter(current_county_climate_rank, to_choose)

            # update to_choose based on sea-level rise data, regardless of preference
            to_choose = self.remove_sea_level_counties(to_choose)

            # if the list is not empty, choose a county
            if to_choose:
                to_move = random.choice(to_choose)
                # if the county is not the current county,  move agent
                if to_move != self.pos:
                    # update county flux counts
                    self.model.county_flux[self.pos] -= 1
                    self.model.county_flux[to_move] += 1
                    # update net migration between counties
                    # (note: if/else statements keep track of which direction
                    # the agent is moving in. if the final value of 'net_migration'
                    # is positive, net migration has been from the larger id 
                    # county to the smaller id county, and vice-versa)
                    if self.pos > to_move:
                        self.model.G[self.pos][to_move]['net_migration'] += 1
                    else:
                        self.model.G[to_move][self.pos]['net_migration'] -= 1
                    # move agent to new county
                    self.model.grid.move_agent(self, to_move)


    ####################################################################################
    #                                   STEP FUNCTIONS                                 #
    ####################################################################################

    def step(self):
        """
        Before migration decision, update agents' attributes.
        """
        self.update_age()
        self.update_income_tenure()
        self.update_network()
        self.calculate_migration_probability()

    def advance(self):
        """
        Once all agents have been updated, agents make migration decision.
        (Simultaneous activation)
        """
        self.make_migration_decision()



####################################################################################
#                               DATA ACCESS FUNCTIONS                              #
####################################################################################
def create_graph():
    """
    Return K_74 complete graph modeling counties, with node/edge attributes
    set to appropriate data.

    Node/edge attributes are stored as dictionaries, which have been
    serialized using pickle.
    """
    # deserialize node data: population, age distribution, income distribution,
    # tenure distribution, climate, sea-level rise, housing
    with open('pickle/node_data_slr_house.pickle', 'rb') as node_data_file:
        node_data = pickle.load(node_data_file)

    # deserialize edge data: distance between each pair of counties
    with open('pickle/distance_dict.pickle', 'rb') as edge_data_file:
        edge_data = pickle.load(edge_data_file)

    # deserialize edge data: "empty" dictionary of migration rates 
    # between each pair of counties
    # (note that all pairs have a value of 0)
    with open('pickle/migration_pair_dict.pickle', 'rb') as mig_data_file:
        migration_edge_data = pickle.load(mig_data_file)

    # create graph and assign attributes
    G = nx.complete_graph(74)
    nx.set_node_attributes(G, node_data)
    nx.set_edge_attributes(G, edge_data, 'distance')
    nx.set_edge_attributes(G, migration_edge_data, 'net_migration')

    return G

def get_population_list():
    """
    Return a list of household counts for each county.
    """
    population_data = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    population_dictionary = population_data.to_dict('list')
    # note that 'total_pop_1000' sets the scale of the model
    return population_dictionary['total_pop_1000']

def get_cumulative_population_list():
    """
    Return a list of cumulative household counts.
    """
    population_data = pd.read_csv('real_data/real_raw_data/raw_totalhousehold.csv')
    cumulative_population = population_data.cumsum(axis=0, skipna=True)
    cumulative_population_dictionary = cumulative_population.to_dict('list')
    # note that 'total_pop_1000' sets the scale of the model
    return cumulative_population_dictionary['total_pop_1000']
