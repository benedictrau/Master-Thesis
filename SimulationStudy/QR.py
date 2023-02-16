import numpy as np
import scipy.stats as stats


def ROP(q, h, p, sim_dur, muh, sigma, c):
    s = (q*h)/(p*c*sim_dur*muh)
    z = stats.norm.ppf(1-s)
    reorder_point = round(muh + z * sigma)

    return reorder_point


def calc_cost(q, r, h, k, muh, sigma, p, c, sim_dur):

    s = (q * h) / (p * c * sim_dur * muh)
    z = stats.norm.ppf(1 - s)
    x = stats.norm.pdf(z) - z * (1 - stats.norm.cdf(z))
    l = sigma * x
    period_cost = h*(r+(q/2)) + (k*muh)/q + (p*l*muh)/q
    #print(period_cost)
    return period_cost


def QR_Calculation(COST_PER_ITEM, COST_PER_ORDER, MEAN_DEMAND_SIZE, SIGMA_DEMAND_SIZE, COST_RATE_HOLDING, COST_RATE_SHORTAGE,
                   BATCH_SIZE_ORDERS, SIM_DURATION, action_space):

    # get the maximum action that can be taken according to the action space
    max_batch_size_multiple = np.argmax(action_space)//2

    best_rop = 0
    best_q = 0
    best_tc = 1000000

    for i in range(max_batch_size_multiple):

        # i+1 since for order quantity of 0 no ROP can be derived
        order_quantity = (i+1) * BATCH_SIZE_ORDERS

        reorder_point = ROP(q=order_quantity, h=COST_RATE_HOLDING, p=COST_RATE_SHORTAGE, sim_dur=SIM_DURATION,
                            muh=MEAN_DEMAND_SIZE, sigma=SIGMA_DEMAND_SIZE, c=COST_PER_ITEM)

        cost = calc_cost(q=order_quantity, r=reorder_point, h=COST_RATE_HOLDING, k=COST_PER_ORDER, muh = MEAN_DEMAND_SIZE,
                         sigma=SIGMA_DEMAND_SIZE, p=COST_RATE_SHORTAGE, c=COST_PER_ITEM, sim_dur=SIM_DURATION)

        #print(f'order quantity: {order_quantity}')
        #print(f'cost: {cost}')
        #print("-----------------")

        if cost < best_tc:
            best_rop = reorder_point
            best_q = order_quantity
            best_tc = cost


    action = (best_q/BATCH_SIZE_ORDERS)*2
    reorder_point = best_rop

    #print(f'action: {action}')
    #print(f'ROP: {reorder_point}')


    return action, reorder_point


result = QR_Calculation(COST_PER_ITEM = 2, COST_PER_ORDER = 10, MEAN_DEMAND_SIZE = 15, SIGMA_DEMAND_SIZE=3, COST_RATE_HOLDING = 0.1,
                        COST_RATE_SHORTAGE=3, BATCH_SIZE_ORDERS=20, SIM_DURATION=200, action_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#result = QR_Calculation(COST_PER_ITEM = 2, COST_PER_ORDER = 10, MEAN_DEMAND_SIZE = 15, SIGMA_DEMAND_SIZE=2, COST_RATE_HOLDING = 0.1,
#                        COST_RATE_SHORTAGE=1.5, BATCH_SIZE_ORDERS=10, SIM_DURATION=200, action_space=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])

#print(result)
