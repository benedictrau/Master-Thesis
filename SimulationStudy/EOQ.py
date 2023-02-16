import math
import numpy as np

def EOQ_Calculation(COST_PER_ORDER, COST_PER_ITEM, MEAN_DEMAND_SIZE, COST_RATE_HOLDING,
                    BATCH_SIZE_ORDERS, SIM_DURATION, action_space):

    # total demand within the considered simulation time
    total_demand = MEAN_DEMAND_SIZE * SIM_DURATION
    # get the maximum action that can be taken according to the action space
    max_batch_size_multiple = np.argmax(action_space)//2

    quantity = math.sqrt((2* total_demand * COST_PER_ORDER) /
                         (COST_PER_ITEM * COST_RATE_HOLDING))


    y_1 = BATCH_SIZE_ORDERS * np.floor(quantity / BATCH_SIZE_ORDERS)
    y_2 = BATCH_SIZE_ORDERS * np.ceil(quantity / BATCH_SIZE_ORDERS)



    # if both candidates are above or equal the max batch_size, the order quantity must
    # equal the maximal possible order quantity
    if y_1 and y_2 >= (max_batch_size_multiple*BATCH_SIZE_ORDERS):
        multiple = max_batch_size_multiple

    else:
        # Calculate the costs of the potential order quantities
        total_c_1 = (COST_PER_ORDER * total_demand) / y_1 + COST_PER_ITEM * total_demand + \
                    y_1 / 2 * COST_RATE_HOLDING

        total_c_2 = (COST_PER_ORDER * total_demand) / y_2 + COST_PER_ITEM * total_demand + \
                    y_2 / 2 * COST_RATE_HOLDING

        # select the order quantity leading to the lowest cost
        if total_c_1 > total_c_2:
            multiple = y_2
        else:
            multiple = y_1

    # since the action is always divided by 2
    action = multiple*2

    return action

#result = EOQ_Calculation(COST_PER_ORDER = 10, COST_PER_ITEM = 2, MEAN_DEMAND_SIZE = 15, COST_RATE_HOLDING = 0.1,
#                     BATCH_SIZE_ORDERS=10, SIM_DURATION=200, action_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#print(result)