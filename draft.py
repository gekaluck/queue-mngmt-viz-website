# data <- combined(Ra,Rp,SIM_TIME,NUM_SERVERS,ADist,PDist,Ra_sd,Rp_sd)
#   Iq_sep <- data.frame(x = data[4][[1]],y = as.numeric(unlist(data[5][[1]])))
#   Iq_sep$mode <- as.factor("Allocation to shortest line")
#   Iq_pool <- data.frame(x = data[6][[1]],y = as.numeric(unlist(data[7][[1]])))
#   Iq_pool$mode <- as.factor("Customer Pooling")
#   Iq_rand <- data.frame(x = data[8][[1]],y = as.numeric(unlist(data[9][[1]])))
#   Iq_rand$mode <- as.factor("Random Allocation")
#   Iq <- rbind(Iq_rand,Iq_sep,Iq_pool)
#   Iq$cat <- as.factor("Iq")

import plotly.express as px
import plotly.figure_factory as ff

from sim import combined

result  = combined(45, 35,500, 4, 'logNormal', 'logNormal', 1, 1)

df = result[0]
print(df)
print(result[1])

df_iq = df.loc[df['cat'] == 'Iq']
fig = px.line(df_iq, 'x', 'y', 'mode')
fig.show()




