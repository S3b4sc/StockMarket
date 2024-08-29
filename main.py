from model_generate.randomForest import saveData

from data.data_import import data
from config import RFContext

if __name__ == '__main__':
    load = data(**RFContext)
    load.importSave()
    
