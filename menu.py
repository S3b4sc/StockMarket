
    
def menu():
    message = '''
    ----------------------------------------------------------------------------
                                CHOOSE DESIRED OPTION
    ----------------------------------------------------------------------------    

        Type the integer of the desired action

        #           ACTION                                  SAVING ROUTE
        1   Generate Random Forest Data                  data/randomForestData/
        2   Generate LSTM Data                           data/lSTMData/
    '''

    option = input(message)

    return option