
    
def menu():
    message = '''
    ----------------------------------------------------------------------------
                                CHOOSE DESIRED OPTION
    ----------------------------------------------------------------------------    

        Type the integer of the desired action

        #           ACTION                                  SAVING ROUTE
        1   Generate Random Forest Data                  data/randomForestData/
        2   Generate LSTM Data                           data/lSTMData/
        3   Train LSTM                                      
    '''

    option = input(message)

    return option