import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# !/usr/bin/env python3.5
# -*- coding: utf-8 -*-

import simpy
import numpy
import pandas as pd

if 'param_show' not in st.session_state:
    st.session_state['param_show'] = 'intro'
if 'simulation' not in st.session_state:
    st.session_state['simulation'] = 'not_ready'

# Adist = Arrival Distribution; Pdist = Process Distribution;
# Ra_sd = Rate of arrival standard deviation; Rp_sd = Rate of processing standard deviation
@st.cache(show_spinner=False)
def combined(ia_t, Tp, SIM_TIME, NUM_SERVERS, ADist, PDist, ia_t_sd, Tp_sd):
    # Defing Normal & LogNormal functions for simulation
    def normalGenerat(a):
        a = numpy.random.normal(a[0], a[1], 1)
        return numpy.absolute(a[0])

    def logNormalGenerate(a):
        a = numpy.random.normal(a[0], a[1], 1)
        if not a[0] == 0:
            return numpy.exp(a[0])

    # RANDOM_SEED = random.randint(1, 99)
    ia_t = float(ia_t)
    Tp = float(Tp)
    SIM_TIME = float(SIM_TIME)
    NUM_SERVERS = int(NUM_SERVERS)

    # Create list and dict for plotting
    customer_pool = {}
    timeStamp_pool = []
    customer_sep = {}
    timeStamp_sep = []
    servers = []
    servers_rand = []
    customer = {}
    timeStamp = []
    Rate = [[Tp, Tp_sd], [ia_t, ia_t_sd]]
    cva = 1
    cvp = 1

    # for different arrival and processing distribution , assiging different paramaters
    if ADist == 'logNormal':
        phiA = numpy.sqrt(ia_t ** 2 + ia_t_sd ** 2)
        phiP = numpy.sqrt(Tp ** 2 + Tp_sd ** 2)
        ARate = [[numpy.log((Tp ** 2) / phiP), numpy.sqrt(numpy.log((phiP ** 2) / (Tp ** 2)))],
                 [numpy.log((ia_t ** 2) / phiA), numpy.sqrt(numpy.log((phiA ** 2) / (ia_t ** 2)))]]
        ADistribution = [logNormalGenerate]
        cva = ia_t_sd

    elif ADist == 'Normal':
        ######
        ARate = Rate
        cva = ia_t_sd
        ADistribution = [normalGenerat]


    elif ADist == 'Fixed':
        ARate = [Tp, ia_t]
        ADistribution = [float]
        cva = 0
    elif ADist == 'Exponential':
        ARate = [Tp, ia_t]
        ADistribution = [numpy.random.exponential]

    if PDist == 'logNormal':
        phiA = numpy.sqrt(ia_t ** 2 + ia_t_sd ** 2)
        phiP = numpy.sqrt(Tp ** 2 + Tp_sd ** 2)
        PRate = [[numpy.log((Tp ** 2) / phiP), numpy.sqrt(numpy.log((phiP ** 2) / (Tp ** 2)))],
                 [numpy.log((ia_t ** 2) / phiA), numpy.sqrt(numpy.log((phiA ** 2) / (ia_t ** 2)))]]
        PDistribution = [logNormalGenerate]
        cvp = Tp_sd

    elif PDist == 'Normal':
        ######
        PRate = Rate
        cvp = Tp_sd
        PDistribution = [normalGenerat]


    elif PDist == 'Fixed':
        PRate = [Tp, ia_t]
        PDistribution = [float]
        cvp = 0
    elif PDist == 'Exponential':
        PRate = [Tp, ia_t]
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
        # customer[name] = [arrival, start, end]
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
        # arrival[name] = [arrival, start, end]
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
        # arrival[name] = [arrival, start, end]
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
            env.process(sim_pool(env_pool, index, server))
            index += 1

    def setup_sep(env, num_machines, processtime, arrivalrate):
        server = Server_sep(env, num_machines, processtime, servers)
        index = 1
        while True:
            yield env.timeout(ADistribution[0](ARate[1]))
            env.process(sim_sep(env_sep, index, server))
            index += 1

    def setup(env, num_machines, processtime, arrivalrate):
        server = Server(env, num_machines, processtime, servers_rand)
        index = 1
        while True:
            yield env.timeout(ADistribution[0](ARate[1]))
            env.process(sim(env, index, server))
            index += 1

    # random.seed(RANDOM_SEED)
    # random.seed(123)
    env_pool = simpy.Environment()
    env_pool.process(setup_pool(env_pool, NUM_SERVERS, Tp, ia_t))
    env_pool.run(until=SIM_TIME)

    env_sep = simpy.Environment()
    env_sep.process(setup_sep(env_sep, NUM_SERVERS, Tp, ia_t))
    env_sep.run(until=SIM_TIME)

    env = simpy.Environment()
    env.process(setup(env, NUM_SERVERS, Tp, ia_t))
    env.run(until=SIM_TIME)

    final_keys_pool = []
    for t in timeStamp_pool:
        if t not in final_keys_pool:
            final_keys_pool.append(t)

    # df = {t:[number_in_queue, utilization_instant,utilization_accumulative, I,Tp,Tq]}
    df_pool = {}
    for t in final_keys_pool:
        q = 0
        inService = 0
        accService = 0
        customerNUM = 0
        totalWait = 0
        customerServed = 0
        for c in customer_pool.keys():
            if customer_pool[c][0] <= t:
                customerNUM += 1
                if len(customer_pool[c]) < 2:
                    totalWait += t - customer_pool[c][0]
                    q += 1
                else:
                    totalWait += customer_pool[c][1] - customer_pool[c][0]
                    if customer_pool[c][1] > t:
                        q += 1

                if len(customer_pool[c]) == 3:
                    if customer_pool[c][2] <= t:
                        accService += customer_pool[c][2] - customer_pool[c][1]
                        customerServed += 1
                    if customer_pool[c][2] > t and customer_pool[c][1] <= t:
                        inService += 1
                        accService += t - customer_pool[c][1]
                        customerServed += 1
                elif len(customer_pool[c]) == 2:
                    if customer_pool[c][1] <= t:
                        accService += t - customer_pool[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_pool[t] = [q]
        if inService / NUM_SERVERS <= 1:
            df_pool[t].append(inService / NUM_SERVERS)
        else:
            df_pool[t].append(1)
        df_pool[t].append(accService / (t * NUM_SERVERS))
        df_pool[t].append(q + inService)
        df_pool[t].append(accService / customerServed)
        df_pool[t].append(totalWait / customerNUM)

    final_keys_sep = []
    for t in timeStamp_sep:
        if t not in final_keys_sep:
            final_keys_sep.append(t)

    # df = {t:[number_in_queue, utilization_instant,utilization_accumulative]}
    df_sep = {}
    for t in final_keys_sep:
        q = 0
        inService = 0
        accService = 0
        customerNUM = 0
        totalWait = 0
        customerServed = 0
        for c in customer_sep.keys():
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
                        accService += customer_sep[c][2] - customer_sep[c][1]
                        customerServed += 1
                    if customer_sep[c][2] > t and customer_sep[c][1] <= t:
                        inService += 1
                        customerServed += 1
                        accService += t - customer_sep[c][1]
                elif len(customer_sep[c]) == 2:
                    if customer_sep[c][1] <= t:
                        accService += t - customer_sep[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_sep[t] = [q]
        if inService / NUM_SERVERS <= 1:
            df_sep[t].append(inService / NUM_SERVERS)
        else:
            df_sep[t].append(1)
        df_sep[t].append(accService / (t * NUM_SERVERS))
        df_sep[t].append(q + inService)
        df_sep[t].append(accService / customerServed)
        df_sep[t].append(totalWait / customerNUM)

    final_keys = []
    for t in timeStamp:
        if t not in final_keys:
            final_keys.append(t)

    df_rand = {}
    for t in final_keys:
        q = 0
        inService = 0
        accService = 0
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
                        accService += customer[c][2] - customer[c][1]
                        customerServed += 1
                    if customer[c][2] > t and customer[c][1] <= t:
                        inService += 1
                        customerServed += 1
                        accService += t - customer[c][1]
                elif len(customer[c]) == 2:
                    if customer[c][1] <= t:
                        accService += t - customer[c][1]
                        inService += 1
                        customerServed += 1
                else:
                    inService = NUM_SERVERS

        df_rand[t] = [q]
        if inService / NUM_SERVERS <= 1:
            df_rand[t].append(inService / NUM_SERVERS)
        else:
            df_rand[t].append(1)
        df_rand[t].append(accService / (t * NUM_SERVERS))
        df_rand[t].append(q + inService)
        df_rand[t].append(accService / customerServed)
        df_rand[t].append(totalWait / customerNUM)

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

    utl_alter_index_pool = []
    for i in range(0, len(utl_y_pool) - 1):
        if utl_y_pool[i] != utl_y_pool[i + 1]:
            utl_alter_index_pool.append(i)
    utl_alter_index_pool = [x + utl_alter_index_pool.index(x) for x in utl_alter_index_pool]

    for i in utl_alter_index_pool:
        utl_x_pool.insert(i + 1, utl_x_pool[i + 1])
        utl_y_pool.insert(i + 1, utl_y_pool[i])

    Iq_alter_index_pool = []
    for i in range(0, len(Iq_y_pool) - 1):
        if Iq_y_pool[i] != Iq_y_pool[i + 1]:
            Iq_alter_index_pool.append(i)
    Iq_alter_index_pool = [x + Iq_alter_index_pool.index(x) for x in Iq_alter_index_pool]

    for i in Iq_alter_index_pool:
        Iq_x_pool.insert(i + 1, Iq_x_pool[i + 1])
        Iq_y_pool.insert(i + 1, Iq_y_pool[i])

    uot_alter_index_pool = []
    for i in range(0, len(uot_y_pool) - 1):
        if uot_y_pool[i] != uot_y_pool[i + 1]:
            uot_alter_index_pool.append(i)
    uot_alter_index_pool = [x + uot_alter_index_pool.index(x) for x in uot_alter_index_pool]

    for i in uot_alter_index_pool:
        uot_x_pool.insert(i + 1, uot_x_pool[i + 1])
        uot_y_pool.insert(i + 1, uot_y_pool[i])

    Tq_alter_index_pool = []
    for i in range(0, len(Tq_y_pool) - 1):
        if Tq_y_pool[i] != Tq_y_pool[i + 1]:
            Tq_alter_index_pool.append(i)
    Tq_alter_index_pool = [x + Tq_alter_index_pool.index(x) for x in Tq_alter_index_pool]

    for i in Tq_alter_index_pool:
        Tq_x_pool.insert(i + 1, Tq_x_pool[i + 1])
        Tq_y_pool.insert(i + 1, Tq_y_pool[i])

    Iq_x_sep = []
    Iq_y_sep = []
    utl_x_sep = []
    utl_y_sep = []
    uot_x_sep = []
    uot_y_sep = []
    Tq_x_sep = []
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

    alter_index_sep = []
    for i in range(0, len(utl_y_sep) - 1):
        if utl_y_sep[i] != utl_y_sep[i + 1]:
            alter_index_sep.append(i)
    alter_index_sep = [x + alter_index_sep.index(x) for x in alter_index_sep]

    for i in alter_index_sep:
        utl_x_sep.insert(i + 1, utl_x_sep[i + 1])
        utl_y_sep.insert(i + 1, utl_y_sep[i])

    Iq_alter_index_sep = []
    for i in range(0, len(Iq_y_sep) - 1):
        if Iq_y_sep[i] != Iq_y_sep[i + 1]:
            Iq_alter_index_sep.append(i)
    Iq_alter_index_sep = [x + Iq_alter_index_sep.index(x) for x in Iq_alter_index_sep]

    for i in Iq_alter_index_sep:
        Iq_x_sep.insert(i + 1, Iq_x_sep[i + 1])
        Iq_y_sep.insert(i + 1, Iq_y_sep[i])

    uot_alter_index_sep = []
    for i in range(0, len(uot_y_sep) - 1):
        if uot_y_sep[i] != uot_y_sep[i + 1]:
            uot_alter_index_sep.append(i)
    uot_alter_index_sep = [x + uot_alter_index_sep.index(x) for x in uot_alter_index_sep]

    for i in uot_alter_index_sep:
        uot_x_sep.insert(i + 1, uot_x_sep[i + 1])
        uot_y_sep.insert(i + 1, uot_y_sep[i])

    Tq_alter_index_sep = []
    for i in range(0, len(Tq_y_sep) - 1):
        if Tq_y_sep[i] != Tq_y_sep[i + 1]:
            Tq_alter_index_sep.append(i)
    Tq_alter_index_sep = [x + Tq_alter_index_sep.index(x) for x in Tq_alter_index_sep]

    for i in Tq_alter_index_sep:
        Tq_x_sep.insert(i + 1, Tq_x_sep[i + 1])
        Tq_y_sep.insert(i + 1, Tq_y_sep[i])

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

    alter_index_rand = []
    for i in range(0, len(utl_y_rand) - 1):
        if utl_y_rand[i] != utl_y_rand[i + 1]:
            alter_index_rand.append(i)
    alter_index_rand = [x + alter_index_rand.index(x) for x in alter_index_rand]

    for i in alter_index_rand:
        utl_x_rand.insert(i + 1, utl_x_rand[i + 1])
        utl_y_rand.insert(i + 1, utl_y_rand[i])

    Iq_alter_index_rand = []
    for i in range(0, len(Iq_y_rand) - 1):
        if Iq_y_rand[i] != Iq_y_rand[i + 1]:
            Iq_alter_index_rand.append(i)
    Iq_alter_index_rand = [x + Iq_alter_index_rand.index(x) for x in Iq_alter_index_rand]

    for i in Iq_alter_index_rand:
        Iq_x_rand.insert(i + 1, Iq_x_rand[i + 1])
        Iq_y_rand.insert(i + 1, Iq_y_rand[i])

    uot_alter_index_rand = []
    for i in range(0, len(uot_y_rand) - 1):
        if uot_y_rand[i] != uot_y_rand[i + 1]:
            uot_alter_index_rand.append(i)
    uot_alter_index_rand = [x + uot_alter_index_rand.index(x) for x in uot_alter_index_rand]

    for i in uot_alter_index_rand:
        uot_x_rand.insert(i + 1, uot_x_rand[i + 1])
        uot_y_rand.insert(i + 1, uot_y_rand[i])

    Tq_alter_index_rand = []
    for i in range(0, len(Tq_y_rand) - 1):
        if Tq_y_rand[i] != Tq_y_rand[i + 1]:
            Tq_alter_index_rand.append(i)
    Tq_alter_index_rand = [x + Tq_alter_index_rand.index(x) for x in Tq_alter_index_rand]

    for i in Tq_alter_index_rand:
        Tq_x_rand.insert(i + 1, Tq_x_rand[i + 1])
        Tq_y_rand.insert(i + 1, Tq_y_rand[i])

    Iq_pool = 0
    Ip_pool = 0
    I_pool = 0
    for i in range(len(df_pool.keys()) - 1):
        Iq_pool += (sorted(df_pool.keys())[i + 1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][
            0] / SIM_TIME
        Ip_pool += (sorted(df_pool.keys())[i + 1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][
            1] * NUM_SERVERS / SIM_TIME
        I_pool += (sorted(df_pool.keys())[i + 1] - sorted(df_pool.keys())[i]) * df_pool[sorted(df_pool.keys())[i]][
            3] / SIM_TIME

    Iq_sep = 0
    Ip_sep = 0
    I_sep = 0
    for i in range(len(df_sep.keys()) - 1):
        Iq_sep += (sorted(df_sep.keys())[i + 1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][
            0] / SIM_TIME
        Ip_sep += (sorted(df_sep.keys())[i + 1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][
            1] * NUM_SERVERS / SIM_TIME
        I_sep += (sorted(df_sep.keys())[i + 1] - sorted(df_sep.keys())[i]) * df_sep[sorted(df_sep.keys())[i]][
            3] / SIM_TIME

    Iq_rand = 0
    Ip_rand = 0
    I_rand = 0
    for i in range(len(df_rand.keys()) - 1):
        Iq_rand += (sorted(df_rand.keys())[i + 1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][
            0] / SIM_TIME
        Ip_rand += (sorted(df_rand.keys())[i + 1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][
            1] * NUM_SERVERS / SIM_TIME
        I_rand += (sorted(df_rand.keys())[i + 1] - sorted(df_rand.keys())[i]) * df_rand[sorted(df_rand.keys())[i]][
            3] / SIM_TIME

    # Making dataframe for plotting in R
    df1 = pd.DataFrame({
        "x": Iq_x_sep,
        "y": Iq_y_sep,
        "mode": "Allocation to shortest line",
        "cat": "Iq"
    })

    df2 = pd.DataFrame({
        "x": Iq_x_pool,
        "y": Iq_y_pool,
        "mode": "Customer Pooling",
        "cat": "Iq"
    })

    df3 = pd.DataFrame({
        "x": Iq_x_rand,
        "y": Iq_y_rand,
        "mode": "Random Allocation",
        "cat": "Iq"
    })

    df4 = pd.DataFrame({
        "x": utl_x_sep,
        "y": utl_y_sep,
        "mode": "Allocation to shortest line",
        "cat": "utl"
    })

    df5 = pd.DataFrame({
        "x": utl_x_pool,
        "y": utl_y_pool,
        "mode": "Customer Pooling",
        "cat": "utl"
    })

    df6 = pd.DataFrame({
        "x": utl_x_rand,
        "y": utl_y_rand,
        "mode": "Random Allocation",
        "cat": "utl"
    })

    df7 = pd.DataFrame({
        "x": uot_x_sep,
        "y": uot_y_sep,
        "mode": "Allocation to shortest line",
        "cat": "uot"
    })

    df8 = pd.DataFrame({
        "x": uot_x_pool,
        "y": uot_y_pool,
        "mode": "Customer Pooling",
        "cat": "uot"
    })

    df9 = pd.DataFrame({
        "x": uot_x_rand,
        "y": uot_y_rand,
        "mode": "Random Allocation",
        "cat": "uot"
    })

    df10 = pd.DataFrame({
        "x": Tq_x_sep,
        "y": Tq_y_sep,
        "mode": "Allocation to shortest line",
        "cat": "Tq"
    })

    df11 = pd.DataFrame({
        "x": Tq_x_pool,
        "y": Tq_y_pool,
        "mode": "Customer Pooling",
        "cat": "Tq"
    })

    df12 = pd.DataFrame({
        "x": Tq_x_rand,
        "y": Tq_y_rand,
        "mode": "Random Allocation",
        "cat": "Tq"
    })

    fdf = pd.merge(df1, df2, how='outer')
    fdf = pd.merge(fdf, df3, how='outer')
    fdf = pd.merge(fdf, df4, how='outer')
    fdf = pd.merge(fdf, df5, how='outer')
    fdf = pd.merge(fdf, df6, how='outer')
    fdf = pd.merge(fdf, df7, how='outer')
    fdf = pd.merge(fdf, df8, how='outer')
    fdf = pd.merge(fdf, df9, how='outer')
    fdf = pd.merge(fdf, df10, how='outer')
    fdf = pd.merge(fdf, df11, how='outer')
    fdf = pd.merge(fdf, df12, how='outer')

    Tq_rand = df_rand[list(df_rand.keys())[-1]][-1]
    Tq_sep = df_sep[list(df_sep.keys())[-1]][-1]
    Tq_pool = df_pool[list(df_pool.keys())[-1]][-1]
    utl_rand = df_rand[list(df_rand.keys())[-1]][2]
    utl_sep = df_sep[list(df_sep.keys())[-1]][2]
    utl_pool = df_pool[list(df_pool.keys())[-1]][2]

    u = Tp / (ia_t * NUM_SERVERS)
    Tq = (ia_t / NUM_SERVERS) * (((u ** (numpy.sqrt(2 * NUM_SERVERS + 2) - 1)) / (1 - u))) * (
                ((cva ** 2) + (cvp ** 2)) / 2)

    I = (Tq + Tp) * (1 / ia_t)

    LIq_pool = (((u ** numpy.sqrt(2 * (NUM_SERVERS + 1))) / (1 - u)) * ((((cva / ia_t) ** 2) + ((cvp / Tp) ** 2)) / 2))
    # Does not take into account whether exponential where the second term = 1 not cva, cvp = 1

    #Need modification for logic reasons R= min(Ra, Rp)
    LIp_pool = (1 / ia_t) * Tp
    LI_pool = (((u ** numpy.sqrt(2 * (NUM_SERVERS + 1))) / (1 - u)) * (((cva ** 2) + (cvp ** 2)) / 2)) + (1 / ia_t) * Tp
    # LIq_pool + LIp_pool ???

    LIq_rs = (((u ** numpy.sqrt(2 * (1 + 1))) / (1 - u)) * (((cva ** 2) + (cvp ** 2)) / 2))
    LIp_rs = u * NUM_SERVERS
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




DISTRIBUTIONS = ["Normal", "logNormal", "Fixed", "Exponential"]

st.set_page_config(layout="wide")
st.sidebar.title("About")
st.sidebar.info("About info About info About info About info About info About info About info About info About info "
                "About info About info About info About info About info About info About info About info "
                "About info About info About info About info About info About info About info About info ")
st.sidebar.title("Credit")
st.sidebar.info("Credit info Credit info Credit info Credit info Credit info Credit info Credit info Credit info "
                "Credit info Credit info Credit info Credit info Credit info Credit info Credit info Credit info "
                "Credit info Credit info Credit info Credit info Credit info Credit info Credit info ")




col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Define Simultaion length", anchor=None)
    length = st.slider('Simulation length', min_value=0, max_value=20000, step=200)
    if length > 0:
        st.session_state['param_show'] = 'show_arr'

with col2:
        if st.session_state['param_show'] == 'show_arr' or st.session_state['param_show'] == 'show_proc':
            st.subheader("Customer arrival parameters", anchor=None)
            q_dist_select = st.selectbox('Interarrival time distribution', DISTRIBUTIONS, key=1)
            a_std = 0
            tp_std = 0
            if q_dist_select == 'Normal':
                a = st.number_input('Average Interarrival time: ', key=11, step=0.1)
                a_std = st.number_input('Standard Deviation: ', key=12, step=0.01)
            elif q_dist_select == 'logNormal':
                a = st.number_input('Mean: ', key=13, step=0.1)
                a_std = st.number_input('Standard Deviation: ', key=14, step=0.01)
            elif q_dist_select == 'Fixed':
                a = st.number_input('Average Interarrival time', key=15, step=0.1)
            elif q_dist_select == 'Exponential':
                a = st.number_input('Average Interarrival time ', key=16, step=0.1)
            if a > 0:
                st.session_state['param_show'] = 'show_proc'

with col3:
        if st.session_state['param_show'] == 'show_proc':
            st.subheader("Define Process parameters", anchor=None)
            workers_num = st.number_input('Number of workers:', key=27, step=1)
            p_dist_select = st.selectbox('Processing time distribution', DISTRIBUTIONS, key=2)
            if p_dist_select == 'Normal':
                tp = st.number_input('Average Processing time: ', key=21)
                tp_std = st.number_input('Standard Deviation: ', key=22)
            elif p_dist_select == 'logNormal':
                tp = st.number_input('Mean: ', key=23)
                tp_std = st.number_input('Standard Deviation: ', key=24)
            elif p_dist_select == 'Fixed':
                tp = st.number_input('Average Processing time', key=25)
            elif p_dist_select == 'Exponential':
                tp = st.number_input('Average Processing time ', key=26)
            if tp > 0:
                st.session_state['simulation'] = 'ready'


with st.container():
    if st.session_state['simulation'] == 'ready':
        start_button = st.button("Start simulation!")
        if start_button:
            st.session_state['param_show'] = 'intro'
            with st.spinner(text='Simulation is in progress...'):
                calculations = combined(a, tp, length, workers_num, q_dist_select, p_dist_select, a_std, tp_std)
                #calculations = [pd.read_csv('output_1.csv')]
                st.session_state['simulation'] = 'finished'
            st.success('Done!')
        else:
            if st.session_state['simulation'] == 'finished':
                st.subheader('Waiting')

with st.container():
    if st.session_state['simulation'] == 'finished':
        st.subheader("Simultaion results: ", anchor=None)
        with st.expander('Instant utilization'):
            utl_df = calculations[0][calculations[0]['cat'] == 'utl']
            fig = go.Figure(layout=go.Layout(width=1200, height=600))
            for mod in utl_df['mode'].unique():
                fig.add_trace(go.Scatter(x=utl_df[utl_df['mode'] == mod]['x'], y=utl_df[utl_df['mode'] == mod]['y'],
                                         mode='lines', name=mod))
                fig.update_xaxes(
                    title_text="Time",
                    title_font={"size": 20},
                    title_standoff=25)

                fig.update_yaxes(
                    title_text="Utilization",
                    title_standoff=25)

            st.write(fig)
        with st.expander('Average utilization'):
            uot_df = calculations[0][calculations[0]['cat'] == 'uot']
            fig1 = go.Figure(layout=go.Layout(width=1200, height=600))
            fig1.update_xaxes(
                title_text="Time",
                title_font={"size": 20},
                title_standoff=25)

            fig1.update_yaxes(
                title_text="Utilization",
                title_standoff=25)
            for mod in uot_df['mode'].unique():
                fig1.add_trace(go.Scatter(x=uot_df[uot_df['mode'] == mod]['x'], y=uot_df[uot_df['mode'] == mod]['y'],
                                         mode='lines', name=mod))
            st.write(fig1)
            st.dataframe(uot_df.rename(columns={'y':'Average utilization'}).groupby('mode').mean()['Average utilization'])

        with st.expander('Customer average waiting time in line'):
            tq_df = calculations[0][calculations[0]['cat'] == 'Tq']
            fig2 = go.Figure(layout=go.Layout(width=1200, height=600))
            fig2.update_xaxes(
                title_text="Time",
                title_font={"size": 20},
                title_standoff=25)

            fig2.update_yaxes(
                title_text="Average waiting time in line",
                title_standoff=25)
            for mod in tq_df['mode'].unique():
                fig2.add_trace(go.Scatter(x=tq_df[tq_df['mode'] == mod]['x'], y=tq_df[tq_df['mode'] == mod]['y'],
                                          mode='lines', name=mod))
            st.write(fig2)
            st.dataframe(tq_df.rename(columns={'y':'Average waiting time'
                                                   ' in line'}).groupby('mode').mean()['Average waiting time in line'])

        with st.expander('Average number of customers in line'):
            iq_df = calculations[0][calculations[0]['cat'] == 'Iq']
            fig3 = go.Figure(layout=go.Layout(width=1200, height=600))
            for mod in iq_df['mode'].unique():
                fig3.add_trace(go.Scatter(x=iq_df[iq_df['mode'] == mod]['x'], y=iq_df[iq_df['mode'] == mod]['y'],
                                          mode='lines', name=mod))
            fig3.update_xaxes(
                title_text="Time",
                title_font={"size": 20},
                title_standoff=25)

            fig3.update_yaxes(
                title_text="Average number of customers in line",
                title_standoff=25)
            st.write(fig3)
            st.dataframe(tq_df.rename(columns={'y':'Average number of customers in line'}).groupby('mode').
                         mean()['Average number of customers in line'])

with st.container():
    if st.session_state['simulation'] == 'finished':
        av_u_string = 'Average utilization:'
        av_iq_string = 'Average number of people in the queue:'
        av_ip_string = 'Average number of people served at a point in time:'
        av_i_string = 'Average number of people in the process:'
        u_string = 'u = Ra/Rp = {}/{} = {} compared to the observed value of {}'
        iq_string = 'Iq = u^sqrt(2*(1+1)) / (1-u)] * [(CVa^2 + CVp^2) / 2] = {} compared to the observed value of {}'
        ip_string = 'Ip = u*c ={} compared to the observed value of {}'
        i_string = 'I = Ip + Iq = {} + {} = {} compared to the observed value of {}'
        st.subheader("Detailed explanation: ", anchor=None)
        st.write("This section estimates service system performance metrics theoretical "
                 "values (using results from queuering theory and the Little's Law) and compares them with the values"
                 "observed from the simulation results.")
        if st.session_state['simulation'] == 'finished':
            with st.expander('Random Assignment'):
                st.write(av_u_string)
                st.code(u_string.format(1/tp, 1/a*workers_num, calculations[1]['u'], calculations[1]['utl_rand']))
                st.write(av_iq_string)
                st.code(iq_string.format(calculations[1]['LIq_rs'], calculations[1]['Iq_rand']))
                st.write(av_ip_string)
                st.code(ip_string.format(calculations[1]['LIp_rs'], calculations[1]['Ip_rand']))
                st.write(av_i_string)
                st.code(i_string.format(calculations[1]['LIp_rs'],calculations[1]['LIq_rs'],
                                        calculations[1]['LI_rs'], calculations[1]['I_rand']))

            with st.expander('Allocation to the shortest line'):
                st.write(av_u_string)
                # st.code(u_string.format(tp, a*workers_num, calculations[1]['u'], calculations[1]['utl_sep']))
                # st.write(av_iq_string)
                # st.code(iq_string.format(calculations[1]['LIq_sep'], calculations[1]['Iq_sep']))
                # st.write(av_ip_string)
                # st.code(ip_string.format(calculations[1]['LIp_sep'], calculations[1]['Ip_sep']))
                # st.write(av_i_string)
                # st.code(i_string.format(calculations[1]['LIp_sep'], calculations[1]['LIq_sep'],
                #                         calculations[1]['LI_sep'], calculations[1]['I_sep']))

            with st.expander('Pooling'):
                st.write(av_u_string)
                st.code(u_string.format(tp, a*workers_num, calculations[1]['u'], calculations[1]['utl_pool']))
                st.write(av_iq_string)
                st.code(iq_string.format(calculations[1]['LIq_pool'],calculations[1]['Iq_pool']))
                st.write(av_ip_string)
                st.code(ip_string.format(calculations[1]['LIp_pool'], calculations[1]['Ip_pool']))
                st.write(av_i_string)
                st.code(i_string.format(calculations[1]['LIp_pool'], calculations[1]['LIq_pool'],
                                        calculations[1]['LI_pool'], calculations[1]['I_pool']))

            with st.expander("Simulation Data"):
                st.dataframe(calculations[0])
                #st.download_button('Download file', calculations[0])

col11, col12 = st.columns(2)
with st.container():
    if st.session_state['simulation'] == 'finished':
        with col11:
            if st.button("Clear simulation results"):
                st.session_state['simulation'] == 'ready'
                calculations = {}












