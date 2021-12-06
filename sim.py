#!/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import simpy
import numpy
import pandas as pd
import streamlit as st


# Adist = Arrival Distribution; Pdist = Process Distribution;
# Ra_sd = Rate of arrival standard deviation; Rp_sd = Rate of processing standard deviation

def combined(Ra,Rp,SIM_TIME,NUM_SERVERS,ADist,PDist,Ra_sd,Rp_sd):

    # Defing Normal & LogNormal functions for simulation
    def normalGenerat(a):
        a = numpy.random.normal(a[0],a[1],1)
        return numpy.absolute(a[0])

    def logNormalGenerate(a):
        a = numpy.random.normal(a[0],a[1],1)
        if not a[0] == 0:
            return numpy.exp(a[0])

    #RANDOM_SEED = random.randint(1, 99)
    Ra = float(Ra)
    Rp = float(Rp)
    SIM_TIME = float(SIM_TIME)
    NUM_SERVERS = int(NUM_SERVERS)

    #Create list and dict for plotting
    customer_pool = {}
    timeStamp_pool = []
    customer_sep = {}
    timeStamp_sep = []
    servers = []
    servers_rand = []
    customer = {}
    timeStamp = []
    Rate = [[Rp,Rp_sd],[Ra,Ra_sd]]
    cva = 1
    cvp = 1


    # for different arrival and processing distribution , assiging different paramaters
    if ADist == 'logNormal':
        phiA = numpy.sqrt(Ra**2 + Ra_sd**2)
        phiP = numpy.sqrt(Rp**2 + Rp_sd**2)
        ARate = [[numpy.log( (Rp**2)/phiP ) , numpy.sqrt( numpy.log((phiP**2)/(Rp**2)))] ,
                [numpy.log( (Ra**2)/phiA ) , numpy.sqrt(numpy.log( (phiA**2)/(Ra**2)))] ]
        ADistribution = [logNormalGenerate]
        cva = Ra_sd

    elif ADist == 'Normal':
        ######
        ARate = Rate
        cva = Ra_sd
        ADistribution = [normalGenerat]


    elif ADist == 'Fixed':
        ARate = [Rp,Ra]
        ADistribution = [float]
        cva = 0
    elif ADist == 'Exponential':
        ARate = [Rp,Ra]
        ADistribution = [numpy.random.exponential]


    if PDist == 'logNormal':
        phiA = numpy.sqrt(Ra**2 + Ra_sd**2)
        phiP = numpy.sqrt(Rp**2 + Rp_sd**2)
        PRate = [[numpy.log( (Rp**2)/phiP ) , numpy.sqrt( numpy.log((phiP**2)/(Rp**2)))] ,
                [numpy.log( (Ra**2)/phiA ) , numpy.sqrt(numpy.log( (phiA**2)/(Ra**2)))] ]
        PDistribution = [logNormalGenerate]
        cvp = Rp_sd

    elif PDist == 'Normal':
        ######
        PRate = Rate
        cvp = Rp_sd
        PDistribution = [normalGenerat]


    elif PDist == 'Fixed':
        PRate = [Rp,Ra]
        PDistribution = [float]
        cvp = 0
    elif PDist == 'Exponential':
        PRate = [Rp,Ra]
        PDistribution = [numpy.random.exponential]


    





    # Define different server type object (Pool / Seperate / Random)
    class Server_pool(object):
        def __init__(self, env, num_servers, processtime):
            self.env = env
            self.machine = simpy.Resource(env, num_servers)
            self.processtime = processtime

        def serve(self, customer):
            yield self.env.timeout(PDistribution[0](PRate[0]))


    class Server_sep(object):
        def __init__(self, env, num_servers, processtime, serverList):
            self.env = env
            for i in range(NUM_SERVERS):
                servers.append(simpy.Resource(env, 1))
            self.serverList = servers
            self.processtime = processtime

        def serve(self, customer):
            yield self.env.timeout(PDistribution[0](PRate[0]))

    class Server(object):
        def __init__(self, env, num_servers, processtime, serverList_rand):
            self.env = env
            for i in range(NUM_SERVERS):
                servers_rand.append(simpy.Resource(env, 1))
            self.serverList_rand = servers_rand
            self.processtime = processtime

        def serve(self, customer):
            yield self.env.timeout(PDistribution[0](PRate[0]))

    def NoInSystem(R):
        return (len(R.queue) + R.count)

    # Define simulation functions, and record every time a customer enters, get service and leave the queue
    def sim_pool(env, name, s):
        #customer[name] = [arrival, start, end]
        customer_pool[name] = [env.now]
        timeStamp_pool.append(env.now)
        with s.machine.request() as request:
            yield request
            customer_pool[name].append(env.now)
            timeStamp_pool.append(env.now)
            yield env.process(s.serve(name))
            customer_pool[name].append(env.now)
            timeStamp_pool.append(env.now)


    def sim_sep(env, name, s):
        #arrival[name] = [arrival, start, end]
        customer_sep[name] = [env.now]
        timeStamp_sep.append(env.now)
        Qlength = [NoInSystem(i) for i in s.serverList]
        for i in range(len(Qlength)):
            if Qlength[i] == 0 or Qlength[i] == min(Qlength):
                choice = i
        with s.serverList[choice].request() as request:
            yield request
            customer_sep[name].append(env.now)
            timeStamp_sep.append(env.now)
            yield env.process(s.serve(name))
            customer_sep[name].append(env.now)
            timeStamp_sep.append(env.now)

    def sim(env, name, s):
        #arrival[name] = [arrival, start, end]
        customer[name] = [env.now]
        timeStamp.append(env.now)
        choice = numpy.random.randint(0, NUM_SERVERS)
        with s.serverList_rand[choice].request() as request:
            yield request
            customer[name].append(env.now)
            timeStamp.append(env.now)
            yield env.process(s.serve(name))
            customer[name].append(env.now)
            timeStamp.append(env.now)



    def setup_pool(env, num_machines, processtime, arrivalrate):
        server = Server_pool(env, num_machines, processtime)
        index = 1
        while True:
            yield env.timeout(ADistribution[0](ARate[1]))
            env.process(sim_pool(env_pool,index, server))
            index += 1

    def setup_sep(env, num_machines, processtime, arrivalrate):
        server = Server_sep(env, num_machines, processtime, servers)
        index = 1
        while True:
            yield env.timeout(ADistribution[0](ARate[1]))
            env.process(sim_sep(env_sep,index, server))
            index += 1

    def setup(env, num_machines, processtime, arrivalrate):
        server = Server(env, num_machines, processtime, servers_rand)
        index = 1
        while True:
            yield env.timeout(ADistribution[0](ARate[1]))
            env.process(sim(env,index, server))
            index += 1



    # random.seed(RANDOM_SEED)
    #random.seed(123)
    env_pool = simpy.Environment()
    env_pool.process(setup_pool(env_pool, NUM_SERVERS, Rp, Ra))
    env_pool.run(until=SIM_TIME)

    env_sep = simpy.Environment()
    env_sep.process(setup_sep(env_sep, NUM_SERVERS, Rp, Ra))
    env_sep.run(until=SIM_TIME)

    env = simpy.Environment()
    env.process(setup(env, NUM_SERVERS, Rp, Ra))
    env.run(until=SIM_TIME)

    final_keys_pool = []
    for t in timeStamp_pool:
        if t not in final_keys_pool:
            final_keys_pool.append(t)

    # df = {t:[number_in_queue, utilization_instant,utilization_accumulative, I,Tp,Tq]}
    df_pool = {}
    for t in final_keys_pool:
        q = 0              # number of people in line?
        inService = 0      # number of people in the system
        total_tp = 0     # cumulated processing time
        customerNUM = 0
        totalWait = 0
        customerServed = 0
        time_diff = 0
        for c in customer_pool.keys(): # ISSUE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


            if customer_pool[c][0] <= t:  # arrival time <= t
                customerNUM += 1  # accept the customer and add to total number of customers
                print('t: ', t)
                print('customer_NUM', customerNUM)
                print(c)
                print(customer_pool[c]) # for each arrived customer [arrival time, start processing time, depart processimg]
                if len(customer_pool[c]) < 2:  # system overload
                    print('len(customer_pool) < 2')
                    print('totalwait: ', totalWait)
                    print('t - customer_pool[c][0]', t - customer_pool[c][0])
                    totalWait += t - customer_pool[c][0]
                    q += 1
                else:  #
                    print('else')
                    print('totalwait: ', totalWait)
                    print('customer_pool[c][1] - customer_pool[c][0]', customer_pool[c][1] - customer_pool[c][0]) # time in line for !!!!!!! if t<=customer_pool[c][1]
                    if customer_pool[c][1] > t:
                        totalWait += t - customer_pool[c][0]
                        q += 1
                    else:
                        totalWait += customer_pool[c][1] - customer_pool[c][0]
                        # no need to update q

                if len(customer_pool[c]) == 3:
                    if customer_pool[c][2] <= t:
                        total_tp += customer_pool[c][2] - customer_pool[c][1]
                        customerServed += 1
                    if customer_pool[c][2] > t and customer_pool[c][1] <= t: # if customer in processing but will not finish processing before t
                        inService += 1
                        total_tp += t - customer_pool[c][1]
                        customerServed += 1
                elif len(customer_pool[c]) == 2:
                    if customer_pool[c][1] <= t:
                        total_tp += t - customer_pool[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_pool[t] = [q]
        if inService/NUM_SERVERS <= 1:
            df_pool[t].append(inService/NUM_SERVERS)
        else:
              df_pool[t].append(1)
        df_pool[t].append(total_tp/(t*NUM_SERVERS))
        df_pool[t].append(q + inService)
        df_pool[t].append(total_tp/customerServed)
        df_pool[t].append(totalWait/customerNUM) #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!11
        print("Average Tq: ", totalWait/customerNUM)
        print("========================================")
    final_keys_sep = []
    for t in timeStamp_sep:
        if t not in final_keys_sep:
            final_keys_sep.append(t)

    # df = {t:[number_in_queue, utilization_instant,utilization_accumulative]}
    df_sep = {}
    for t in final_keys_sep:
        q = 0
        inService = 0
        total_tp = 0
        customerNUM = 0
        totalWait = 0
        customerServed = 0
        for c in customer_sep.keys(): # ISSUE
            if customer_sep[c][0] <= t:
                customerNUM += 1
                if len(customer_sep[c]) < 2:
                    totalWait += t - customer_sep[c][0]
                    q += 1
                else:
                    totalWait += customer_sep[c][1] - customer_sep[c][0]
                    if customer_sep[c][1] > t:
                        q += 1

                if len(customer_sep[c]) == 3:
                    if customer_sep[c][2] <= t:
                        total_tp += customer_sep[c][2] - customer_sep[c][1]
                        customerServed += 1
                    if customer_sep[c][2] > t and customer_sep[c][1] <= t:
                        inService += 1
                        customerServed += 1
                        total_tp += t - customer_sep[c][1]
                elif len(customer_sep[c]) == 2:
                    if customer_sep[c][1] <= t:
                        total_tp += t - customer_sep[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_sep[t] = [q]
        if inService/NUM_SERVERS <= 1:
            df_sep[t].append(inService/NUM_SERVERS)
        else:
              df_sep[t].append(1)
        df_sep[t].append(total_tp/(t*NUM_SERVERS))
        df_sep[t].append(q + inService)
        df_sep[t].append(total_tp/customerServed)
        df_sep[t].append(totalWait/customerNUM)
        # Total wait / num of customers who departed the line

    final_keys = []
    for t in timeStamp:
        if t not in final_keys:
            final_keys.append(t)

    df_rand = {}
    for t in final_keys:
        q = 0
        inService = 0
        total_tp = 0
        customerNUM = 0
        totalWait = 0
        customerServed = 0
        for c in customer.keys():
            if customer[c][0] <= t:
                customerNUM += 1
                if len(customer[c]) < 2:
                    q += 1
                    totalWait += t - customer[c][0]
                else:
                    totalWait += customer[c][1] - customer[c][0]
                    if customer[c][1] > t:
                        q += 1

                if len(customer[c]) == 3:
                    if customer[c][2] <= t:
                        total_tp += customer[c][2] - customer[c][1]
                        customerServed += 1
                    if customer[c][2] > t and customer[c][1] <= t:
                        inService += 1
                        customerServed += 1
                        total_tp += t - customer[c][1]
                elif len(customer[c]) == 2:
                    if customer[c][1] <= t:
                        total_tp += t - customer[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_rand[t] = [q]
        if inService/NUM_SERVERS <= 1:
            df_rand[t].append(inService/NUM_SERVERS)
        else:
              df_rand[t].append(1)
        df_rand[t].append(total_tp/(t*NUM_SERVERS))
        df_rand[t].append(q + inService)
        df_rand[t].append(total_tp/customerServed)
        df_rand[t].append(totalWait/customerNUM)

    Iq_x_pool = []
    Iq_y_pool = []
    utl_x_pool = []
    utl_y_pool = []
    uot_x_pool = []
    uot_y_pool = []
    Tq_x_pool = []
    Tq_y_pool = []
    for a in sorted(list(df_pool.keys())):
        utl_x_pool.append(a)
        Iq_x_pool.append(a)
        uot_x_pool.append(a)
        Tq_x_pool.append(a)
        utl_y_pool.append(df_pool[a][1])
        Iq_y_pool.append(df_pool[a][0])
        uot_y_pool.append(df_pool[a][2])
        Tq_y_pool.append(df_pool[a][-1])


    utl_alter_index_pool =[]
    for i in range(0, len(utl_y_pool)-1):
        if utl_y_pool[i] != utl_y_pool[i+1]:
            utl_alter_index_pool.append(i)
    utl_alter_index_pool = [x+ utl_alter_index_pool.index(x) for x in utl_alter_index_pool]

    for i in utl_alter_index_pool:
        utl_x_pool.insert(i+1, utl_x_pool[i+1])
        utl_y_pool.insert(i+1, utl_y_pool[i])

    Iq_alter_index_pool =[]
    for i in range(0, len(Iq_y_pool)-1):
        if Iq_y_pool[i] != Iq_y_pool[i+1]:
            Iq_alter_index_pool.append(i)
    Iq_alter_index_pool = [x+ Iq_alter_index_pool.index(x) for x in Iq_alter_index_pool]

    for i in Iq_alter_index_pool:
        Iq_x_pool.insert(i+1, Iq_x_pool[i+1])
        Iq_y_pool.insert(i+1, Iq_y_pool[i])

    uot_alter_index_pool = []
    for i in range(0, len(uot_y_pool)-1):
        if uot_y_pool[i] != uot_y_pool[i+1]:
            uot_alter_index_pool.append(i)
    uot_alter_index_pool = [x+ uot_alter_index_pool.index(x) for x in uot_alter_index_pool]

    for i in uot_alter_index_pool:
        uot_x_pool.insert(i+1, uot_x_pool[i+1])
        uot_y_pool.insert(i+1, uot_y_pool[i])
        
    Tq_alter_index_pool = []
    for i in range(0, len(Tq_y_pool)-1):
        if Tq_y_pool[i] != Tq_y_pool[i+1]:
            Tq_alter_index_pool.append(i)
    Tq_alter_index_pool = [x+ Tq_alter_index_pool.index(x) for x in Tq_alter_index_pool]

    for i in Tq_alter_index_pool:
        Tq_x_pool.insert(i+1, Tq_x_pool[i+1])
        Tq_y_pool.insert(i+1, Tq_y_pool[i])

    Iq_x_sep = []
    Iq_y_sep = []
    utl_x_sep = []
    utl_y_sep = []
    uot_x_sep = []
    uot_y_sep = []
    Tq_x_sep =[]
    Tq_y_sep = []
    for a in sorted(list(df_sep.keys())):
        utl_x_sep.append(a)
        Iq_x_sep.append(a)
        uot_x_sep.append(a)
        Tq_x_sep.append(a)
        utl_y_sep.append(df_sep[a][1])
        Iq_y_sep.append(df_sep[a][0])
        uot_y_sep.append(df_sep[a][2])
        Tq_y_sep.append(df_sep[a][-1])


    alter_index_sep =[]
    for i in range(0, len(utl_y_sep)-1):
        if utl_y_sep[i] != utl_y_sep[i+1]:
            alter_index_sep.append(i)
    alter_index_sep = [x+ alter_index_sep.index(x) for x in alter_index_sep]

    for i in alter_index_sep:
        utl_x_sep.insert(i+1, utl_x_sep[i+1])
        utl_y_sep.insert(i+1, utl_y_sep[i])

    Iq_alter_index_sep =[]
    for i in range(0, len(Iq_y_sep)-1):
        if Iq_y_sep[i] != Iq_y_sep[i+1]:
            Iq_alter_index_sep.append(i)
    Iq_alter_index_sep = [x+ Iq_alter_index_sep.index(x) for x in Iq_alter_index_sep]

    for i in Iq_alter_index_sep:
        Iq_x_sep.insert(i+1, Iq_x_sep[i+1])
        Iq_y_sep.insert(i+1, Iq_y_sep[i])

    uot_alter_index_sep = []
    for i in range(0, len(uot_y_sep)-1):
        if uot_y_sep[i] != uot_y_sep[i+1]:
            uot_alter_index_sep.append(i)
    uot_alter_index_sep = [x+ uot_alter_index_sep.index(x) for x in uot_alter_index_sep]

    for i in uot_alter_index_sep:
        uot_x_sep.insert(i+1, uot_x_sep[i+1])
        uot_y_sep.insert(i+1, uot_y_sep[i])
        
    Tq_alter_index_sep = []
    for i in range(0, len(Tq_y_sep)-1):
        if Tq_y_sep[i] != Tq_y_sep[i+1]:
            Tq_alter_index_sep.append(i)
    Tq_alter_index_sep = [x+ Tq_alter_index_sep.index(x) for x in Tq_alter_index_sep]

    for i in Tq_alter_index_sep:
        Tq_x_sep.insert(i+1, Tq_x_sep[i+1])
        Tq_y_sep.insert(i+1, Tq_y_sep[i])

    Iq_x_rand = []
    Iq_y_rand = []
    utl_x_rand = []
    utl_y_rand = []
    uot_x_rand = []
    uot_y_rand = []
    Tq_x_rand = []
    Tq_y_rand = []
    for a in sorted(list(df_rand.keys())):
        utl_x_rand.append(a)
        Iq_x_rand.append(a)
        uot_x_rand.append(a)
        Tq_x_rand.append(a)
        utl_y_rand.append(df_rand[a][1])
        Iq_y_rand.append(df_rand[a][0])
        uot_y_rand.append(df_rand[a][2])
        Tq_y_rand.append(df_rand[a][-1])


    alter_index_rand =[]
    for i in range(0, len(utl_y_rand)-1):
        if utl_y_rand[i] != utl_y_rand[i+1]:
            alter_index_rand.append(i)
    alter_index_rand = [x+ alter_index_rand.index(x) for x in alter_index_rand]

    for i in alter_index_rand:
        utl_x_rand.insert(i+1, utl_x_rand[i+1])
        utl_y_rand.insert(i+1, utl_y_rand[i])

    Iq_alter_index_rand =[]
    for i in range(0, len(Iq_y_rand)-1):
        if Iq_y_rand[i] != Iq_y_rand[i+1]:
            Iq_alter_index_rand.append(i)
    Iq_alter_index_rand = [x+ Iq_alter_index_rand.index(x) for x in Iq_alter_index_rand]

    for i in Iq_alter_index_rand:
        Iq_x_rand.insert(i+1, Iq_x_rand[i+1])
        Iq_y_rand.insert(i+1, Iq_y_rand[i])

    uot_alter_index_rand = []
    for i in range(0, len(uot_y_rand)-1):
        if uot_y_rand[i] != uot_y_rand[i+1]:
            uot_alter_index_rand.append(i)
    uot_alter_index_rand = [x+ uot_alter_index_rand.index(x) for x in uot_alter_index_rand]

    for i in uot_alter_index_rand:
        uot_x_rand.insert(i+1, uot_x_rand[i+1])
        uot_y_rand.insert(i+1, uot_y_rand[i])
        
    Tq_alter_index_rand = []
    for i in range(0, len(Tq_y_rand)-1):
        if Tq_y_rand[i] != Tq_y_rand[i+1]:
            Tq_alter_index_rand.append(i)
    Tq_alter_index_rand = [x+ Tq_alter_index_rand.index(x) for x in Tq_alter_index_rand]

    for i in Tq_alter_index_rand:
        Tq_x_rand.insert(i+1, Tq_x_rand[i+1])
        Tq_y_rand.insert(i+1, Tq_y_rand[i])

    Iq_pool = 0
    Ip_pool = 0
    I_pool  = 0
    for i in range(len(df_pool.keys())-1):
        Iq_pool += (sorted(df_pool.keys())[i+1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][0]/SIM_TIME
        Ip_pool += (sorted(df_pool.keys())[i+1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][1]*NUM_SERVERS/SIM_TIME
        I_pool  += (sorted(df_pool.keys())[i+1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][3]/SIM_TIME

    Iq_sep = 0
    Ip_sep = 0
    I_sep  = 0
    for i in range(len(df_sep.keys())-1):
        Iq_sep += (sorted(df_sep.keys())[i+1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][0]/SIM_TIME
        Ip_sep += (sorted(df_sep.keys())[i+1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][1]*NUM_SERVERS/SIM_TIME
        I_sep  += (sorted(df_sep.keys())[i+1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][3]/SIM_TIME

    Iq_rand = 0
    Ip_rand = 0
    I_rand  = 0
    for i in range(len(df_rand.keys())-1):
        Iq_rand += (sorted(df_rand.keys())[i+1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][0]/SIM_TIME
        Ip_rand += (sorted(df_rand.keys())[i+1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][1]*NUM_SERVERS/SIM_TIME
        I_rand  += (sorted(df_rand.keys())[i+1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][3]/SIM_TIME


    # Making dataframe for plotting in R
    df1 = pd.DataFrame({
        "x":Iq_x_sep,
        "y":Iq_y_sep,
        "mode":"Allocation to shortest line",
        "cat":"Iq"
        })

    df2 = pd.DataFrame({
        "x":Iq_x_pool,
        "y":Iq_y_pool,
        "mode":"Customer Pooling",
        "cat":"Iq"
        })

    df3 = pd.DataFrame({
        "x":Iq_x_rand,
        "y":Iq_y_rand,
        "mode":"Random Allocation",
        "cat":"Iq"
        })

    df4 = pd.DataFrame({
        "x":utl_x_sep,
        "y":utl_y_sep,
        "mode":"Allocation to shortest line",
        "cat":"utl"
        })

    df5 = pd.DataFrame({
        "x":utl_x_pool,
        "y":utl_y_pool,
        "mode":"Customer Pooling",
        "cat":"utl"
        })

    df6 = pd.DataFrame({
        "x":utl_x_rand,
        "y":utl_y_rand,
        "mode":"Random Allocation",
        "cat":"utl"
        })

    df7 = pd.DataFrame({
        "x":uot_x_sep,
        "y":uot_y_sep,
        "mode":"Allocation to shortest line",
        "cat":"uot"
        })

    df8 = pd.DataFrame({
        "x":uot_x_pool,
        "y":uot_y_pool,
        "mode":"Customer Pooling",
        "cat":"uot"
        })

    df9 = pd.DataFrame({
        "x":uot_x_rand,
        "y":uot_y_rand,
        "mode":"Random Allocation",
        "cat":"uot"
        })

    df10 = pd.DataFrame({
        "x":Tq_x_sep,
        "y":Tq_y_sep,
        "mode":"Allocation to shortest line",
        "cat":"Tq"
        })

    df11 = pd.DataFrame({
        "x":Tq_x_pool,
        "y":Tq_y_pool,
        "mode":"Customer Pooling",
        "cat":"Tq"
        })

    df12 = pd.DataFrame({
        "x":Tq_x_rand,
        "y":Tq_y_rand,
        "mode":"Random Allocation",
        "cat":"Tq"
        })

    fdf = pd.merge(df1,df2, how ='outer')
    fdf = pd.merge(fdf,df3, how ='outer')
    fdf = pd.merge(fdf,df4, how ='outer')
    fdf = pd.merge(fdf,df5, how ='outer')
    fdf = pd.merge(fdf,df6, how ='outer')
    fdf = pd.merge(fdf,df7, how ='outer')
    fdf = pd.merge(fdf,df8, how ='outer')
    fdf = pd.merge(fdf,df9, how ='outer')
    fdf = pd.merge(fdf,df10, how ='outer')
    fdf = pd.merge(fdf,df11, how ='outer')
    fdf = pd.merge(fdf,df12, how ='outer')

    Tq_rand = df_rand[list(df_rand.keys())[-1]][-1]
    Tq_sep = df_sep[list(df_sep.keys())[-1]][-1]
    Tq_pool = df_pool[list(df_pool.keys())[-1]][-1]
    utl_rand = df_rand[list(df_rand.keys())[-1]][2]
    utl_sep = df_sep[list(df_sep.keys())[-1]][2]
    utl_pool = df_pool[list(df_pool.keys())[-1]][2]
    
    u = Rp/(Ra*NUM_SERVERS)
    Tq = (Ra/NUM_SERVERS)*(((u**(numpy.sqrt(2*NUM_SERVERS+2)-1))/(1-u)))*(((cva**2)+(cvp**2))/2)

    I = (Tq+Rp)*(1/Ra)
    
    LIq_pool = (((u**numpy.sqrt(2*(NUM_SERVERS+1)))/(1-u)) * ((((cva)**2)+((cvp)**2))/2))
    #LIq_pool = (((u**numpy.sqrt(2*(NUM_SERVERS+1)))/(1-u)) * ((((cva/Ra)**2)+((cvp/Rp)**2))/2))
    LIp_pool = (1/Ra)*Rp
    LI_pool = (((u**numpy.sqrt(2*(NUM_SERVERS+1)))/(1-u)) * (((cva**2)+(cvp**2))/2)) + (1/Ra)*Rp
    
    LIq_rs = (((u**numpy.sqrt(2*(1+1)))/(1-u)) * (((cva**2)+(cvp**2))/2))
    LIp_rs = u*NUM_SERVERS
    LI_rs = LIq_rs + LIp_rs

    # return fdf ,"{:.4f}".format(Iq_rand),"{:.4f}".format(Iq_sep),"{:.4f}".format(Iq_pool) ,"{:.4f}".format(Tq_rand),
    # "{:.4f}".format(Tq_sep),"{:.4f}".format(Tq_pool),"{:.4f}".format(utl_rand),"{:.4f}".format(utl_sep),
    # "{:.4f}".format(utl_pool),"{:.4f}".format(Ip_rand),"{:.4f}".format(Ip_sep),"{:.4f}".format(Ip_pool) ,
    # "{:.4f}".format(I_rand),"{:.4f}".format(I_sep),"{:.4f}".format(I_pool), "{:.4f}".format(LIq_pool),
    # "{:.4f}".format(LIp_pool),"{:.4f}".format(LI_pool), "{:.4f}".format(LIq_rs),"{:.4f}".format(LIp_rs),
    # "{:.4f}".format(LI_rs),"{:.4f}".format(u)

    result_values = {
        'Iq_rand': round(Iq_rand, 4),
        'Iq_sep': round(Iq_sep, 4),
        'Iq_pool': round(Iq_pool, 4),
        'Tq_rand': round(Tq_rand, 4),
        'Tq_sep': round(Tq_sep, 4),
        'Tq_pool': round(Tq_pool, 4),
        'utl_rand': round(utl_rand, 4),
        'utl_sep': round(utl_sep, 4),
        'utl_pool': round(utl_pool, 4),
        'Ip_rand': round(Ip_rand, 4),
        'Ip_sep': round(Ip_sep, 4),
        'Ip_pool': round(Ip_pool, 4),
        'I_rand': round(I_rand, 4),
        'I_sep': round(I_sep, 4),
        'I_pool': round(I_pool, 4),
        'LIq_pool': round(LIq_pool, 4),
        'LIp_pool': round(LIp_pool, 4),
        'LI_pool': round(LI_pool, 4),
        'LIq_rs': round(LIq_rs, 4),
        'LIp_rs': round(LIp_rs, 4),
        'LI_rs': round(LI_rs, 4),
        'u': round(u, 4)
    }
    return fdf, result_values


# result = combined(3, 5, 20000, 1, 'Exponential', 'Exponential', 0.5, 0.5)
# result[0].to_csv("output.csv")
# print(result[1])

# Ra,Rp,SIM_TIME,NUM_SERVERS,ADist,PDist,Ra_sd,Rp_sd

# Ra = 3 # Rp = 5 T = 20000

# {'Iq_rand': 1219.2213, 'Iq_sep': 1331.3227, 'Iq_pool': 1219.4286, 'Tq_rand': 3722.813, 'Tq_sep':
# 4019.0873, 'Tq_pool': 3717.2033, 'utl_rand': 0.9999, 'utl_sep': 0.9996, 'utl_pool': 0.999, ' q':
# 0.9998, 'Ip_sep': 0.9995, 'Ip_pool': 0.999, 'I_rand': 1220.221, 'I_sep': 1332.3222, 'I_pool': 1220.4275,
# 'LIq_pool': -0.3148, 'LIp_pool': 1.6667, 'LI_pool': -2.5, 'LIq_rs': -4.1667, 'LIp_rs': 1.6667,
# 'LI_rs': -2.5, 'u': 1.6667}

# R = min(Ra, Rp) = Ra = 3 -> a = 1/3
# I= R*T = 1/3
# Rp = 5 -> Tp = 1/5


result = combined(2, 5, 50, 1, 'Fixed', 'Fixed', 0, 0)
result[0].to_csv("output_1.csv")
print(result[1])


# {'Iq_rand': 29.0796, 'Iq_sep': 7.8275, 'Iq_pool': 7.56, 'Tq_rand': 73.2023, 'Tq_sep': 19.5298, 'Tq_pool': 18.646,
#  'utl_rand': 0.8896, 'utl_sep': 0.8975, 'utl_pool': 0.9073, 'Ip_rand': 3.5636, 'Ip_sep': 3.5896, 'Ip_pool': 3.6292,
#  'I_rand': 32.645, 'I_sep': 11.4171, 'I_pool': 11.1892, 'LIq_pool': 0.6176, 'LIp_pool': 3.6, 'LI_pool': 10.7664,
#  'LIq_rs': 8.1, 'LIp_rs': 3.6, 'LI_rs': 11.7, 'u': 0.9}

# a = 2.5 Tp = 9 -> Ra = 0.4 Rp = 4 * 1/9 = 0.4444
#