import simpy
import numpy as np

class InventorySystem:

    def __init__(self,
                 mean_demand_size = 4,
                 sigma_demand_size = 0.5,
                 start_inventory = 20,
                 render_env = False,
                 sim_duration = 200,
                 time_step = 1,
                 demand_deviation_boundary = 0.7,
                 invisible_demand_size = 0.7,
                 batch_size = 5,
                 deviation_direction = 0.5
                 ):

        # Define global variables
        self.system_level = start_inventory
        self.actual_level = start_inventory
        self.state = dict()
        self.state['system_level'] = 0
        self.state['last_stock_count'] = 0

        self.render_env = render_env
        self.sim_duration = sim_duration
        self.time_step = time_step
        self.next_time_stop = 0
        self.actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


        self.demand_deviation_boundary = demand_deviation_boundary
        self.invisible_demand_size = invisible_demand_size
        self.deviation_direction = deviation_direction
        self.batch_size = batch_size
        self.last_change = 0
        # Random parameters
        self.start_inventory = start_inventory
        self.mean_demand_size = mean_demand_size
        self.sigma_demand_size = sigma_demand_size
        self.last_stock_check = 0


    def order(self, action):
        # Function deciding whether parts should be ordered or not including the arrival process

        # Round action down (7 means 3) to ensure that pairs like (0,1), (2,3), etc. are belonging to the same order quantity
        action = action // 2
        quantity = action*self.batch_size
        # Increase stock
        self.system_level += quantity
        self.actual_level += quantity


    def demand(self):

        # VISIBLE DEMAND
        # Calculate demand size
        global invisible_demand
        self.demand_size = np.around(max(np.random.normal(self.mean_demand_size, self.sigma_demand_size), 0), 0)
        # Increase total demand
        # Update holding costs
        self.last_change = self.env.now
        # Reduce system level according to the demand
        if self.actual_level >= self.demand_size:
            self.system_level -= self.demand_size
            self.actual_level -= self.demand_size
            # If we don't run out of stock then increase KPI satisfied demand
        elif self.actual_level < self.demand_size:
            # Decrease system and actual level according to the remaining parts in stock
            # More than what's on stock can't be sold
            self.system_level -= self.actual_level
            self.actual_level = 0

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


    def check_inventory(self, action):
        # if modulo == 1: count stock (meaning for actions 1,3,5,7,9,11 a stock count will be conducted)
        if (action % 2) == 1:
            self.system_level = self.actual_level
            self.last_stock_check = 0


    def render(self):
        """Display current state"""
        print (f"Weekday: {self.env.now}, ", end = '')
        print (f"System Level: {self.system_level}, ", end = '')
        print (f"Actual Level: {self.actual_level}, ", end = '')


    def reset(self):
        """Reset environment"""
        # Initialise SimPy environment
        self.env = simpy.Environment()
        # Reset the different starting parameters
        self.system_level = self.start_inventory
        self.actual_level = self.start_inventory
        self.next_time_stop = 0
        self.last_stock_check = 0
        self.last_change = 0
        self.state.clear()
        #self.state['pending_orders'] = 0
        #self.state['pending_parts'] = 0
        #self.state['system_level'] = 0
        #self.state['fraction_of_satisfied_orders'] = 0
        #self.state['fraction_of_satisfied_demand'] = 0

        # Get and return observation
        observations = self.get_observation()
        actual_stock_class, actual_stock = self.get_stock_class()
        return (observations, actual_stock_class, actual_stock)


    def get_observation(self):

        self.state['system_level'] = self.system_level
        self.state['last_stock_count'] = self.last_stock_check

        # Return observation results
        observations = [v for k,v in self.state.items()]

        return observations


    def get_stock_class(self):

        stock = self.actual_level

        stock_class = stock//5
        if stock_class > 20:
            stock_class = 20

        return stock_class, stock


    def step(self, action):
        # Increase last stock check
        self.last_stock_check += 1
        # Demand arrives
        self.demand()
        # Stock gets counted
        self.check_inventory(action)
        # Make a step in the simulation and run the simulation until this time
        # Replenishment order arrives
        self.order(action)
        # Make a step in the simulation and run the simulation until this time
        self.next_time_stop += self.time_step
        self.env.run(until=self.next_time_stop)
        # Check whether end of simulation (terminal state) is reached (based on sim time)
        terminal = True if self.env.now >= self.sim_duration else False
        # Calculate costs and set last change to now
        self.last_change = self.env.now
        # Get reward and observation
        actual_stock_class, actual_stock = self.get_stock_class()
        observations = self.get_observation()
        # Render environment if requested
        if self.render_env:
            self.render()
        # Get dict
        info = dict()

        # Return tuple of observations, reward, terminal, info
        return (observations, actual_stock_class, actual_stock, terminal, info)


