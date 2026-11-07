"""Hand-specified synthetic mechanisms, not estimated medical effects."""
import itertools


def confounded():
    # Z=0 favorable context; X assigned more often there. X harms Y in both Z.
    return {'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]},
            {'name':'V','probabilities':[.2]*5},{'name':'W','probabilities':[.1]*10}],
            'nodes':[{'name':'Z','states':2,'inputs':['U'],'table':[0,1]},
                     {'name':'X','states':2,'inputs':['Z','V'],'table':[int(v<(4 if z==0 else 1)) for z,v in itertools.product(range(2),range(5))]},
                     {'name':'Y','states':2,'inputs':['Z','X','W'],'table':[int(w<[[9,8],[3,2]][z][x]) for z,x,w in itertools.product(range(2),range(2),range(10))]}]}


def observational_twins():
    return {'direct':{'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]}],
            'nodes':[{'name':'X','states':2,'inputs':['U'],'table':[0,1]},
                     {'name':'Y','states':2,'inputs':['X'],'table':[0,1]}]},
            'common':{'version':1,'exogenous':[{'name':'U','probabilities':[.5,.5]}],
            'nodes':[{'name':'X','states':2,'inputs':['U'],'table':[0,1]},
                     {'name':'Y','states':2,'inputs':['U'],'table':[0,1]}]}}
