def summary(model):
    # Prints a summary of the given model
    cols = ["Layer", "Type", "Input Size", "Kernel Size", "# Filters", "Nonlinearity", "Pooling", "Stride", "Size", "Output Shape", "Parameters"]
    summary = {}
    nlayers = []
 
    for i in range(len(model.layers)):
        if "conv" in model.layers[i].name or "dense" in model.layers[i].name:
            nlayers += [i]

    for i in nlayers:
        nlayer = i+1
        summary[nlayer] = {c:"" for c in cols[1:]}

        summary[nlayer][cols[1]] = model.layers[i].name
        summary[nlayer][cols[2]] = str(model.layers[i].input_shape)
        summary[nlayer][cols[9]] = str(model.layers[i].output_shape)
        if "conv" in model.layers[i].name or "dense" in model.layers[i].name:
            summary[nlayer][cols[5]] = get_nonlinearity(model,i)
        weights = model.layers[i].get_weights()
        if "dense" in summary[nlayer][cols[1]]:
            units = weights[0].shape[0] * weights[0].shape[1]
            bias = weights[1].shape[0]
            summary[nlayer][cols[10]] = units + bias

        if "conv" in summary[nlayer][cols[1]]:
            summary[nlayer][cols[3]] = (model.layers[i].nb_row, model.layers[i].nb_col)
            summary[nlayer][cols[4]] = model.layers[i].nb_filter    
            summary[nlayer][cols[7]] = model.layers[i].subsample
            pooling_name, size = get_pooling(model,i)
            summary[nlayer][cols[6]] = pooling_name
            summary[nlayer][cols[8]] = size
            
            kernel = weights[0].shape[0] * weights[0].shape[1]
            channels = weights[0].shape[2]
            filters = weights[0].shape[3]
            bias = weights[1].shape[0]
            summary[nlayer][cols[10]] = kernel * channels * filters + bias

    for c in cols:
        print "%-20s" %(c) + "|",
    print ""
    n = 0
    for l in sorted(summary):
        n += 1
        print "%-20i" %(n) +"|",
        for c in cols[1:]:
            print "%-20s" % (str(summary[l][c])) +"|",
        print ""

def get_nonlinearity(model,i):
    if model.layers[i].activation.__name__ != "linear":
        return str(model.layers[i].activation.__name__)
    for n in range(i+1, len(model.layers)):
        if "conv" in model.layers[n].name or "dense" in model.layers[n].name or "flat" in model.layers[n].name:
            return ""
        elif "act" in model.layers[n].name:
            return str(model.layers[n].activation.__name__)

def get_pooling(model,i):
    for n in range(i+1, len(model.layers)):
        if "conv" in model.layers[n].name or "dense" in model.layers[n].name or "flat" in model.layers[n].name:
            return ""
        elif "pool" in model.layers[n].name:
            return str(model.layers[n].name), model.layers[n].pool_size
