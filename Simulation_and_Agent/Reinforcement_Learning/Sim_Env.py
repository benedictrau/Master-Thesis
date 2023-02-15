import simpy
import numpy as np


class InventorySystem:

    def __init__(self,
                 mean_demand_size = 6,
                 sigma_demand_size = 0.5,
                 start_inventory = 100,
                 render_env = False,
                 sim_duration = 30,
                 time_step = 1,
                 cost_per_item = 2,
                 cost_per_order = 20,
                 cost_rate_shortage = 0.11,
                 cost_rate_holding = 0.1,
                 cost_inventory_check = 20,
                 demand_deviation_boundary = 0.7,
                 invisible_demand_size=0.6,
                 batch_size = 10,
                 deviation_direction = 0.5
                 ):

        # Define global variables
        self.system_level = start_inventory
        self.actual_level = start_inventory
        self.actual_inventory_last_period = start_inventory

        self.state = dict()
        self.state['system_level'] = 0
        self.state['last_stock_count'] = 0
        self.state['actual_level'] = 0

        self.KPI = dict()
        self.KPI['fraction_of_satisfied_demand'] = 0
        self.KPI['fraction_of_satisfied_orders'] = 0
        self.KPI['satisfied_demand'] = 0
        self.KPI['missed_demand'] = 0
        self.KPI['satisfied_orders'] = 0
        self.KPI['missed_orders'] = 0

        self.render_env = render_env
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.next_time_stop = 0
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

        self.demand_deviation_boundary = demand_deviation_boundary
        self.invisible_demand_size = invisible_demand_size
        self.batch_size = batch_size
        self.deviation_direction = deviation_direction
        self.cost_per_item = cost_per_item
        self.cost_per_order = cost_per_order
        self.cost_rate_holding = cost_rate_holding
        self.cost_rate_shortage = cost_rate_shortage
        self.cost_inventory_check = cost_inventory_check
        self.last_change = 0
        self.total_costs = 0
        self.period_cost = 0
        self.order_costs = 0
        self.holding_cost = 0
        self.shortage_cost = 0
        self.inventory_checking_cost = 0

        # Random parameters
        self.start_inventory = start_inventory
        self.mean_demand_size = mean_demand_size
        self.sigma_demand_size = sigma_demand_size

        self.last_stock_check = 0
        self.satisfied_orders = 0
        self.unmet_orders = 0
        self.satisfied_demand = 0
        self.unmet_demands = 0
        self.total_demand = 0
        self.fraction_of_satisfied_orders = 0
        self.fraction_of_satisfied_demand = 0
        self.demand_size = 0

        self.action_size = 12
        self.observation_space = 21


    def order(self, action):
        # Function deciding whether parts should be ordered or not including the arrival process

        # Round action down (7 means 3) to ensure that pairs like (0,1), (2,3), etc. are belonging to the same order quantity
        action = action // 2
        quantity = action*self.batch_size


        # Update Order Costs
        if action > 0:
            self.order_costs = (self.cost_per_order + self.cost_per_item * quantity)
        else:
            self.order_costs += 0

        # Increase stock
        self.system_level += quantity
        self.actual_level += quantity
        #print(str(quantity) + " parts have been arrived")


    def demand(self):

        # VISIBLE DEMAND
        # Calculate demand size
        global invisible_demand
        #print(f'system level: {self.system_level}')
        #print(f'actual level: {self.actual_level}')
        self.demand_size = np.around(max(np.random.normal(self.mean_demand_size, self.sigma_demand_size), 0), 0)
        #print(f'regular demand size: {self.demand_size}')
        # Increase total demand
        self.total_demand += self.demand_size
        # Reduce system level according to the demand
        if self.actual_level >= self.demand_size:
            self.system_level -= self.demand_size
            self.actual_level -= self.demand_size
            # If we don't run out of stock then increase KPI satisfied demand
            self.satisfied_demand += self.demand_size
            self.satisfied_orders += 1
        elif self.actual_level < self.demand_size:
            # Decrease system and actual level according to the remaining parts in stock
            self.shortage_cost = (self.demand_size - self.actual_level) * self.cost_rate_shortage * self.cost_per_item
            self.satisfied_demand += self.actual_level
            #print("unmet demand")
            self.unmet_demands += (self.demand_size - self.actual_level)
            self.unmet_orders += 1
            # More than what's on stock can't be sold
            self.system_level -= self.actual_level
            self.actual_level = 0

        #print("----- after regular demand ----")
        #print(f'system level: {self.system_level}')
        #print(f'actual level: {self.actual_level}')

        # INVISIBLE DEMAND
        random_deviation = np.random.uniform(0, 1)
        # Demand deviation boundary determines the fraction of demands with deviations
        if random_deviation >= self.demand_deviation_boundary:
            random_demand = np.random.uniform(0, self.invisible_demand_size)
            invisible_demand = max(np.round(self.demand_size*random_demand, 0), 1)


            deviation_direction = np.random.uniform(0, 1)
            if deviation_direction >= self.deviation_direction:
                invisible_demand = invisible_demand * -1
            else:
                invisible_demand = invisible_demand
            if self.actual_level != 0:
                # If actual level is bigger then the invisible demand the whole invisible demand can be fulfilled
                if self.actual_level >= invisible_demand:
                    self.actual_level -= invisible_demand
                # If the invisible demand is bigger then the stock level, only the remaining parts can be consumed
                elif self.actual_level < invisible_demand:
                    self.actual_level = 0
                # If the stock is already empty, no invisible demand can occur
                #print(f'invisible demand: {invisible_demand}')
            #print("----- after invisible demand ----")
            #print(f'system level: {self.system_level}')
            #print(f'actual level: {self.actual_level}')


    def calculate_holding_costs(self):
        # Calculate the holding cost depending on the cost for holding inventory, the time and the stock level
        holding_cost = self.cost_rate_holding * (self.env.now - self.last_change) * (self.actual_level+self.actual_inventory_last_period)/2
        self.holding_cost += np.round(holding_cost, 0)
        self.actual_inventory_last_period = self.actual_level


    def reward(self):
        # calculate the reward (costs that occur in the respective period)
        self.period_cost = self.shortage_cost + self.holding_cost + self.order_costs + self.inventory_checking_cost
        self.total_costs += self.period_cost
        loss = -abs(self.period_cost)
        return loss


    def check_inventory(self, action):
        # if modulo == 1: count stock (meaning action 1,3,5,7,9,11
        if (action % 2) == 1:
            #print("inventory checked")
            self.system_level = self.actual_level
            self.inventory_checking_cost = self.cost_inventory_check
            self.last_stock_check = 1
        else:
            self.inventory_checking_cost = 0
            #print("inventory not checked")


    def action_allowed(self, action):
        # Check if the action that was taken is allowed or not
        if action not in self.actions:
            raise ValueError('Requested action not in list of allowed actions')


    def render(self):
        """Display current state"""
        print(f"Weekday: {self.env.now}, ", end = '')
        print(f"System Level: {self.system_level}, ", end = '')
        print(f"Actual Level: {self.actual_level}, ", end = '')
        print(f"Order Costs: {self.order_costs}, ", end = '')
        print(f"Holding Costs: {self.holding_cost}, ", end = '')
        print(f"Shortage Costs: {self.shortage_cost}, ", end = '')
        print(f'Cost Inventory Check: {self.inventory_checking_cost}  ', end = '')
        print(f"Period Costs: {self.period_cost}, ", end='')
        print(f"Total Costs: {-self.total_costs}")
        #print([v for k,v in self.state.items()])


    def reset(self):
        """Reset environment"""
        # Initialise SimPy environment
        self.env = simpy.Environment()
        # Reset the different starting parameters
        self.system_level = self.start_inventory
        self.actual_level = self.start_inventory
        self.actual_inventory_last_period = self.start_inventory
        self.next_time_stop = 0
        self.total_costs = 0
        self.period_cost = 0
        self.order_costs = 0
        self.holding_cost = 0
        self.shortage_cost = 0
        self.inventory_checking_cost = 0
        self.last_stock_check = 0
        self.last_change = 0
        self.state.clear()
        self.fraction_of_satisfied_orders = 0
        self.fraction_of_satisfied_demand = 0
        self.satisfied_demand = 0
        self.satisfied_orders = 0
        self.unmet_demands = 0
        self.unmet_orders = 0

        # Get and return observation
        observations = self.get_observation()
        return observations


    def get_observation(self):
        # Update dict
        #self.state['pending_orders'] = self.pending_orders
        self.state['system_level'] = self.system_level
        self.state['last_stock_count'] = self.last_stock_check
        self.state['actual_level'] = self.actual_level

        # Return observation results
        observations = [v for k,v in self.state.items()]
        return observations


    def get_KPI(self):
        # Calculation of service level
        if (self.satisfied_orders + self.unmet_orders) > 0:
            self.fraction_of_satisfied_orders = np.round(self.satisfied_orders /
                                                         (self.satisfied_orders + self.unmet_orders), 2)
        if (self.satisfied_demand + self.unmet_demands) > 0:
            self.fraction_of_satisfied_demand = np.round(self.satisfied_demand/
                                                     (self.satisfied_demand + self.unmet_demands), 2)

        self.KPI['fraction_of_satisfied_demand'] = self.fraction_of_satisfied_demand
        self.KPI['fraction_of_satisfied_orders'] = self.fraction_of_satisfied_orders
        self.KPI['satisfied_demand'] = self.satisfied_demand
        self.KPI['missed_demand'] = self.unmet_demands
        self.KPI['satisfied_orders'] = self.satisfied_orders
        self.KPI['missed_orders'] = self.unmet_orders

        KPI = [v for k, v in self.KPI.items()]
        return KPI


    def step(self, action):
        # Increase last stock check
        self.last_stock_check += 1
        # Check whether this action is allowed or not
        self.action_allowed(action)
        # Replenishment order arrives
        self.order(action)
        # Demand arrives
        self.demand()
        # Stock gets counted
        self.check_inventory(action)
        # Make a step in the simulation and run the simulation until this time
        self.next_time_stop += self.time_step
        self.env.run(until=self.next_time_stop)
        # Check whether end of simulation (terminal state) is reached (based on sim time)
        terminal = True if self.env.now >= self.sim_duration else False
        # Calculate costs and set last change to now
        self.calculate_holding_costs()
        self.last_change = self.env.now
        # Get reward and observation
        reward = self.reward()
        observations = self.get_observation()
        KPI = self.get_KPI()
        # Render environment if requested
        if self.render_env:
            self.render()
        # Get dict
        info = dict()

        # set holding and order cost to 0 before next step. They should just consider the costs of one sim step
        self.holding_cost = 0
        self.order_costs = 0
        self.shortage_cost = 0
        self.inventory_checking_cost = 0


        # Return tuple of observations, reward, terminal, info
        return (observations, reward, terminal, info, KPI)
