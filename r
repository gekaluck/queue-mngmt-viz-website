# Making the plot for Iq
plotMaker_Iq <- function(Ra,Rp,NUM_SERVERS,df){
  df <- df %>% arrange()
  p <- ggplotly(ggplot(data=df%>%filter(cat == 'Iq'))+
                  aes(x = as.numeric(x), y = as.numeric(y), color = mode)+
                  theme_set(theme_stata())+
                  geom_line()+
                  labs(x = 'Time', y = 'Number of Customer in Queue',
                       # 'Number of Customer in Queue','\n',
                       title = paste('Avg. I.A time: ',Ra, ', Avg. processing time: ',Rp,',  # of Servers: ',NUM_SERVERS))+
                  theme(plot.title = element_text(size = rel(1),hjust=0.5),
                        legend.title = element_blank(),
                        legend.background = element_rect(fill = "white",
                                                         colour = "white"),
                        axis.title=element_text(face = "bold", size = rel(1)),
                        plot.background = element_rect(fill = "white"),
                        text = element_text(family = "Arial")) +
                  scale_colour_manual(values =c('#8ac926','#1982c4','#f8961e'))
  ) %>%
    layout(legend = list(
      orientation = "h",
      y=1.1
    )
    )

  return(ggplotly(p))
}

# Making the Plot for uot
plotMaker_uot <- function(Ra,Rp,NUM_SERVERS,df){
  df <- df %>% arrange()
  p <- ggplotly(ggplot(data=df%>%filter(cat == 'uot'))+
                  aes(x = as.numeric(x), y = as.numeric(y), color = mode)+
                  theme_set(theme_stata())+
                  geom_line()+
                  labs(x = 'Time', y = 'Accumulative Utilization',
                       # 'Average Utilization','\n',
                       title = paste('Avg. I.A time: ',Ra, ', Avg. processing time: ',Rp,',  # of Servers: ',NUM_SERVERS))+
                  theme(plot.title = element_text(size = rel(1),hjust=0.5),
                        legend.title = element_blank(),
                        legend.background = element_rect(fill = "white",
                                                         colour = "white"),
                        axis.title=element_text(face = "bold", size = rel(1)),
                        plot.background = element_rect(fill = "white"),
                        text = element_text(family = "Arial")) +
                  scale_colour_manual(values =c('#8ac926','#1982c4','#f8961e'))
  ) %>%
    layout(legend = list(
      orientation = "h",
      y=1.1
    )
    )

  return(ggplotly(p))
}

# Making plot for Tq
plotMaker_Tq <- function(Ra,Rp,NUM_SERVERS,df){
  df <- df %>% arrange()
  p <- ggplotly(ggplot(data=df%>%filter(cat == 'Tq'))+
                  aes(x = as.numeric(x), y = as.numeric(y), color = mode)+
                  theme_set(theme_stata())+
                  geom_line()+
                  labs(x = 'Time', y = 'Average Wait Time in Queue',
                       # 'Average Utilization','\n',
                       title = paste('Avg. I.A time: ',Ra, ', Avg. processing time: ',Rp,',  # of Servers: ',NUM_SERVERS))+
                  theme(plot.title = element_text(size = rel(1),hjust=0.5),
                        legend.title = element_blank(),
                        legend.background = element_rect(fill = "white",
                                                         colour = "white"),
                        axis.title=element_text(face = "bold", size = rel(1)),
                        plot.background = element_rect(fill = "white"),
                        text = element_text(family = "Arial")) +
                  scale_colour_manual(values =c('#8ac926','#1982c4','#f8961e'))
  ) %>%
    layout(legend = list(
      orientation = "h",
      y=1.1
    )
    )

  return(ggplotly(p))
}

# Making plot for Utilization
plotMaker_utl <- function(Ra,Rp,NUM_SERVERS,df){
  df <- df %>% arrange()
  p <- ggplotly(ggplot(data=df%>%filter(cat == 'utl'))+
                  aes(x = as.numeric(x), y = as.numeric(y), color = mode)+
                  theme_set(theme_stata())+
                  geom_line()+
                  labs(x = 'Time', y = 'Instantaneous Utilization',
                       # 'Average Utilization','\n',
                       title = paste('Avg. I.A time: ',Ra, ', Avg. processing time: ',Rp,',  # of Servers: ',NUM_SERVERS))+
                  theme(plot.title = element_text(size = rel(1),hjust=0.5),
                        legend.title = element_blank(),
                        legend.background = element_rect(fill = "white",
                                                         colour = "white"),
                        axis.title=element_text(face = "bold", size = rel(1)),
                        plot.background = element_rect(fill = "white"),
                        text = element_text(family = "Arial")) +
                  scale_colour_manual(values =c('#8ac926','#1982c4','#f8961e'))
  ) %>%
    layout(legend = list(
      orientation = "h",
      y=1.1
    )
    )

  return(ggplotly(p))
}



# Making the frontend pages with bs4page package
shiny::shinyApp(
  ui = bs4DashPage(
    old_school = FALSE,
    sidebar_mini = TRUE,
    sidebar_collapsed = FALSE,
    controlbar_collapsed = FALSE,
    controlbar_overlay = FALSE,
    title = "Service Simulation",
    navbar = bs4DashNavbar(),
    sidebar = bs4DashSidebar(
      skin = "dark",
      status = "primary",
      title = "Service Simulation",
      brandColor = "primary",
      url = "https://lxqing.shinyapps.io/LittleLaw/",
      src = "https://www.numbersupermarket.co.uk/images/icons/Call-Queue.png",
      elevation = 3,
      opacity = 0.8,

      bs4SidebarMenu(
        bs4SidebarHeader("Dashboard Overview"),
        bs4SidebarMenuItem(
          "About",
          tabName = "home",
          icon = "home"
        ),
        bs4SidebarHeader("Simulation"),
        bs4SidebarMenuItem(
          "Setup and Results",
          tabName = "item1",
          icon = "chart-line"
        ),

        bs4SidebarHeader("Appendix"),
        bs4SidebarMenuItem(
          "Credit",
          tabName = "about",
          icon = "info"
        )

      )
    ),


    footer = bs4DashFooter(),
    body = bs4DashBody(

      useShinyjs(),

      tags$style(type="text/css",
                 ".shiny-output-error { visibility: hidden; }",
                 ".shiny-output-error:before { visibility: hidden; }"),

      bs4TabItems(

        bs4TabItem(
          tabName = "home",
          # textOutput('py_config'),
          # tableOutput('table')
          bs4Card(
            title = tags$h3("Explanation", style = "text-align:center"),
            width = '100%',
            status = 'light',
            closable = FALSE,
            tags$p("This interactive Service System Simulation is designed solely for educational purposes. This dashboard illustrates the impact of different kinds of variabilities, i.e. variability in customer's arrival pattern and variability in processing time of each customer on performance metrics of a service system. ")
          ),
          bs4Card(
            title = tags$h3("Demo", style = "text-align:center"),
            width = '100%',
            status = 'light',
            closable = FALSE,
            fluidRow(width=12,
                     column(width = 3),
                     column(width = 6,tags$iframe(width="560", height="315", src="https://www.youtube.com/embed/akldMDvodhk", frameborder="0", allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture", allowfullscreen=NA)),
                     column(width = 3))
          )
        ),

        # Making the input area
        bs4TabItem(
          tabName = "item1",
          bs4Card(
            title = tags$h3("Simulation Setup", style = "text-align:center"),
            width = '100%',
            status = 'dark',
            closable = FALSE,
            fluidRow(width=12,
                     column(width = 4,
                            bs4Card(
                              title = tags$h6("Simulation Length", style = "text-align:center"),
                              width = '100%',
                              status = 'success',
                              # collapsed = TRUE,
                              closable = FALSE,
                              numericInput('SIMTIME','Simulation Length','')
                            )),
                     column(width = 4,
                            bs4Card(
                              title = tags$h6("Customer Arrival Parameters", style = "text-align:center"),
                              width = '100%',
                              status = 'primary',
                              # collapsed = TRUE,
                              closable = FALSE,
                              selectInput("ADist", "Customer Interarrival Time Distribution:",
                                          c('Exponential','Normal','Fixed','LogNormal'),
                                          multiple = FALSE,
                                          selected =  ''),
                              numericInput('Ra','Average Interarrival time',''),
                              numericInput('Ra_sd','Std.Dev. of Inter-Arrival time','')
                            )),
                     column(width = 4,
                            bs4Card(
                              title = tags$h6("Customer Process Parameters", style = "text-align:center"),
                              width = '100%',
                              status = 'warning',
                              # collapsed = TRUE,
                              closable = FALSE,
                              selectInput("PDist", "Customer Processing Time Distribution:",
                                          c('Exponential','Normal','Fixed','LogNormal'),
                                          multiple = FALSE,
                                          selected =  ''),
                              numericInput('Rp','Customer Average Processing time',''),
                              numericInput('NUM_SERVERS','Number of Servers',''),

                              numericInput('Rp_sd','Std.Dev. of Processing time','')
                            ))

            )
          ),


          # Making the output area
          bs4Card(width = 12,
                  title = tags$h3("Simulation Output Section", style = "text-align:center"),
                  status = 'secondary',
                  closable = FALSE,
                  collapsed = TRUE,
                  bs4Card(
                    title = tags$h4("Servers Instantaneous Utilization", style = "text-align:center"),
                    closable = FALSE,
                    collapsible = TRUE,
                    status = 'light',
                    width = '100%',
                    plotlyOutput('utl')
                  ),

                  bs4Card(
                    title = tags$h4("Server Average Utilization", style = "text-align:center"),
                    closable = FALSE,
                    collapsible = TRUE,
                    status = 'light',
                    width = '100%',
                    plotlyOutput('uot'),
                    fluidRow(width=12,
                             column(width = 3),
                             column(width = 6,tableOutput('df_uot')),
                             column(width = 3))


                  ),

                  bs4Card(
                    title = tags$h4("Customer Average Wait Time in Line(Queue)", style = "text-align:center"),
                    closable = FALSE,
                    collapsible = TRUE,
                    status = 'light',
                    width = '100%',
                    plotlyOutput('Tq'),
                    fluidRow(width=12,
                             column(width = 3),
                             column(width = 6,tableOutput('df_Tq')),
                             column(width = 3))

                  ),

                  bs4Card(
                    title = tags$h4("Total Number of Customers in line(Queue)", style = "text-align:center"),
                    closable = FALSE,
                    collapsible = TRUE,
                    status = 'light',
                    width = '100%',
                    plotlyOutput('Iq'),
                    fluidRow(width=12,
                             column(width = 3),
                             column(width = 6,tableOutput('df_Iq')),
                             column(width = 3))

                  ),

                  bs4Card(
                    title = tags$h4("Calculation of service system performance metrics theoretical values and comparing them with values observed from the simulation", style = "text-align:center"),
                    closable = FALSE,
                    collapsible = TRUE,
                    status = 'secondary',
                    width = '100%',
                    bs4Card(
                      title = tags$h5("Explanation", style = "text-align:center"),
                      width = '100%',
                      status = 'light',
                      collapsed = FALSE,
                      closable = FALSE,
                      tags$h6("Explanation: this section tries to calculate service system performance metrics theoretical values(using results from queuering theory and Little's Law) and compares them with values ovserved directly from the simulation.")),

                    bs4Card(
                      title = tags$h5("Random Assignment", style = "text-align:center"),
                      width = '100%',
                      status = 'light',
                      collapsed = FALSE,
                      closable = FALSE,

                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('u_rand')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Iq_rand')))
                      ),

                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Ip_rand')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('I_rand')))
                      )

                    ),
                    bs4Card(
                      title = tags$h5("Allocation to shortest line", style = "text-align:center"),
                      width = '100%',
                      status = 'white',
                      collapsed = FALSE,
                      closable = FALSE,

                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('u_sep')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Iq_sep')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Ip_sep')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('I_sep')))
                      )

                    ),
                    bs4Card(
                      title = tags$h5("Customer Pooling", style = "text-align:center"),
                      width = '100%',
                      status = 'light',
                      collapsed = FALSE,
                      closable = FALSE,

                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('u_pool')))
                               ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Iq_pool')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('Ip_pool')))
                      ),
                      fluidRow(width=12,
                               column(width = 12,
                                      tags$h6(textOutput('I_pool')))
                      )

                    )

                  )



                ),



        ),

        bs4TabItem(
          tabName = "about",

          fluidRow(width=12,
                   column(width = 6,
                            bs4UserCard(
                              src = "https://carey.jhu.edu/sites/default/files/styles/large/public/2019-10/Naser%20Nikandish.jpg?itok=atdkYK6S",
                              status = "info",
                              width = "100%",
                              title = "Dr. Naser Nikandish",
                              elevation = 2,
                              tags$a(href = "mailto:nnikand1@jhu.edu","Email: nnikand1@jhu.edu")
                            )
                          ),

                   column(width = 6,
                          bs4UserCard(
                            src = "yl.jpg",
                            status = "info",
                            title = "Yuqin Lin",
                            width = "100%",
                            elevation = 2,
                            tags$a(href = "mailto:ylin129@jhu.edu","Email: ylin129@jhu.edu")
                          ))

                   ),




        )



      )

    )

  ),
  server = function(input, output) {

    observeEvent(input$ADist,{
      if (input$ADist == 'Normal' | input$ADist == 'LogNormal') {
        shinyjs::show(id = "Ra_sd")
      }else{
        shinyjs::hide(id = "Ra_sd")
      }
    })

    observeEvent(input$PDist,{
      if (input$PDist == 'Normal' | input$PDist == 'LogNormal') {
        shinyjs::show(id = "Rp_sd")
      }else{
        shinyjs::hide(id = "Rp_sd")
      }
    })

    dataInput <- reactive({
      # dfMaker(input$Ra,input$Rp,input$SIMTIME,input$NUM_SERVERS,input$ADist,input$PDist,input$Ra_sd,input$Rp_sd)
      df <- combined(input$Ra,input$Rp,input$SIMTIME,input$NUM_SERVERS,input$ADist,input$PDist,input$Ra_sd,input$Rp_sd)
      df[[1]]$mode <- as.factor(df[[1]]$mode)
      df[[1]]$cat <- as.factor(df[[1]]$cat)
      df[[1]]$mode <- factor(df[[1]]$mode, levels = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  ))
      df
    })



    theme_set(theme_stata())

    output$utl <- renderPlotly({
      df <- dataInput()
      # df$mode <- factor(df$mode, levels = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  ))
      plotMaker_utl(input$Ra,input$Rp,input$NUM_SERVERS,df[[1]])
    })
    #
    output$uot <- renderPlotly({
      df <- dataInput()
      # df$mode <- factor(df$mode, levels = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  ))
      plotMaker_uot(input$Ra,input$Rp,input$NUM_SERVERS,df[[1]])
    })
    #
    output$Tq <- renderPlotly({
      df <- dataInput()
      # df$mode <- factor(df$mode, levels = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  ))
      plotMaker_Tq(input$Ra,input$Rp,input$NUM_SERVERS,df[[1]])
    })

    output$Iq <- renderPlotly({
      df <- dataInput()
      plotMaker_Iq(input$Ra,input$Rp,input$NUM_SERVERS,df[[1]])
    })

    output$df_uot <- renderTable({
      df <- dataInput()
      df_uot <- data.frame("Queue Discipline" = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  )
                           ,"Average Utilization" = c(df[[8]],df[[9]],df[[10]]))

      return(df_uot)

    }
    ,colnames = TRUE
    ,width = '100%'
    ,striped = TRUE
    ,hover = TRUE)

    output$df_Tq <- renderTable({
      df <- dataInput()
      df_Tq <- data.frame("Queue Discipline" = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  )
                           ,"Average Utilization" = c(df[[5]],df[[6]],df[[7]]))

      return(df_Tq)

    }
    ,colnames = TRUE
    ,width = '100%'
    ,striped = TRUE
    ,hover = TRUE)

    output$df_Iq <- renderTable({
      df <- dataInput()
      df_Iq <- data.frame("Queue Discipline" = c( "Random Allocation" ,"Allocation to shortest line","Customer Pooling"  )
                          ,"Average Utilization" = c(df[[2]],df[[3]],df[[4]]))

      return(df_Iq)

    }
    ,colnames = TRUE
    ,width = '100%'
    ,striped = TRUE
    ,hover = TRUE)

    output$u_rand <- renderText({
      df <- dataInput()
      t <- paste('u = Ra/Rp =',input$Rp,'/',as.numeric(input$Ra)*as.numeric(input$NUM_SERVERS),df[[23]],sep=" ")
      return(t)
    })
    output$Iq_rand <- renderText({
      df <- dataInput()
      t <- paste('Iq = u^sqrt(2*(1+1)) / (1-u)] * [(CVa^2 + CVp^2) / 2] =',df[[17]],'compare to observed one at',df[[2]],sep=" ")
      return(t)
    })
    output$Ip_rand <- renderText({
      df <- dataInput()
      t <- paste('Ip = u*c =',df[[23]],'*',input$NUM_SERVERS,'=',df[[18]],'compare to observed value at',df[[11]],sep=" ")
      return(t)
    })
    output$I_rand <- renderText({
      df <- dataInput()
      t <- paste('I = Ip + Iq =', df[[17]],'+',df[[18]],'=',df[[19]],'compare to observed value at',df[[14]],sep=" ")
      return(t)
    })

    output$u_sep <- renderText({
      df <- dataInput()
      t <- paste('u = Ra/Rp =',input$Rp,'/',as.numeric(input$Ra)*as.numeric(input$NUM_SERVERS),df[[23]],sep=" ")
      return(t)
    })
    output$Iq_sep <- renderText({
      df <- dataInput()
      t <- paste('Iq = u^sqrt(2*(1+1)) / (1-u)] * [(CVa^2 + CVp^2) / 2] =',df[[17]],'compare to observed one at',df[[3]],sep=" ")
      return(t)
    })
    output$Ip_sep <- renderText({
      df <- dataInput()
      t <- paste('Ip = u*c =',df[[23]],'*',input$NUM_SERVERS,'=',df[[18]],'compare to observed value at',df[[12]],sep=" ")
      return(t)
    })
    output$I_sep <- renderText({
      df <- dataInput()
      t <- paste('I = Ip + Iq =', df[[17]],'+',df[[18]],'=',df[[19]],'compare to observed value at',df[[15]],sep=" ")
      return(t)
    })

    output$u_pool <- renderText({
      df <- dataInput()
      t <- paste('u = Ra/Rp =',df[[23]],sep=" ")
      return(t)
    })
    output$Iq_pool <- renderText({
      df <- dataInput()
      t <- paste('Iq = u^sqrt(2*(c+1)) / (1-u)] * [(CVa^2 + CVp^2) / 2] =',df[[20]],'compare to observed one at',df[[4]],sep=" ")
      return(t)
    })
    output$Ip_pool <- renderText({
      df <- dataInput()
      t <- paste('Ip = u*c =',df[[23]],'*',input$NUM_SERVERS,'=',df[[21]],'compare to observed value at',df[[13]],sep=" ")
      return(t)
    })
    output$I_pool <- renderText({
      df <- dataInput()
      t <- paste('I = Ip + Iq =', df[[20]],'+',df[[21]],'=',df[[22]],'compare to observed value at',df[[16]],sep=" ")
      return(t)
    })
  }
)
