import torch 
from sympy import lambdify
from sklearn.preprocessing import MinMaxScaler
from sympy import Symbol
import numpy as np

def traslate_sentence_from_numbers(numerical_expression, model, device, max_len = 66):
    model.eval()
    scaler = MinMaxScaler()
    xxx_n = scaler.fit_transform(numerical_expression.reshape(-1,1))
    #xxx_n = np.sin(np.expand_dims(av(support),0))
    x_new = torch.from_numpy(xxx_n.T).float().to(device)
    with torch.no_grad():
        encoder_conved, encoder_combined = model.encoder(x_new) 
        trg_indexes = [12]
        for i in range(max_len):
            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)
            pred_token = output.argmax(2)[:,-1].item()
            trg_indexes.append(pred_token)
            if pred_token == 13:
                break
    return trg_indexes, attention, xxx_n

def translate_sentence_0(expression, model, device,support, max_len = 66):

    model.eval()
    x = Symbol('x')

    av = lambdify(x,expression)
    xxx_n = np.expand_dims(av(support),0)/1
    scaler = MinMaxScaler()
    xxx_n = scaler.fit_transform(xxx_n.T)
    #xxx_n = np.sin(np.expand_dims(av(support),0))
    x_new = torch.from_numpy(xxx_n.T).float().to(device)

    with torch.no_grad():
        print('here')
        encoder_conved, encoder_combined = model.encoder(x_new)

        
        trg_indexes = [12]
        for i in range(max_len):

            trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)



            with torch.no_grad():
                output, attention = model.decoder(trg_tensor, encoder_conved, encoder_combined)




            pred_token = output.argmax(2)[:,-1].item()



            trg_indexes.append(pred_token)

            if pred_token == 13:
                break
    
    return trg_indexes, attention, xxx_n,av