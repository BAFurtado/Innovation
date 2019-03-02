from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.batchrunner import BatchRunner
import numpy as np
import pandas as pd


class Firm(Agent):

    def __init__(self, unique_id, model, numberknowntasks, probautomated, expectedoutputq, rdstock):
        super().__init__(unique_id, model)
        self.knowntasks =  self.model.taskset.loc[np.random.choice(range(len(self.model.taskset.index)),
                                                                   numberknowntasks,replace=False)]
        self.knowntasks['KNOWLEDGE'] = np.random.choice((0.90,0),numberknowntasks, p=[probautomated,1-probautomated])
        self.expectedoutputq= expectedoutputq
        self.rdstock = rdstock
        self.opnetincomehist = []
        self.outputq=0
        self.robots=0
        self.employees=0
        self.outputaddedval=0
        self.revenues=0
        self.opnetincome=0
        self.productinnovation = 0
        self.processinnovation = 0
        self.everythingautomated = 0
        self.belowthresholdaddedval = 0
        self.attemptprocessinn = 0

    def settasks(self):
        self.knowntasks = self.knowntasks.sort_values('ADDEDVAL',ascending=False)
        self.currenttasks = self.knowntasks.iloc[:self.model.nperformedtasks]
        self.allautomatedtasks = self.knowntasks[self.knowntasks['KNOWLEDGE'] >= self.knowntasks['SKILLREQ']]
        self.allmanualtasks = self.knowntasks[self.knowntasks['KNOWLEDGE'] < self.knowntasks['SKILLREQ']]
        self.allautomatabletasks = self.allmanualtasks[self.allmanualtasks['SKILLREQ'] <= 0.90]
        self.currentautomatedtasks = self.currenttasks.loc[self.allautomatedtasks.index.intersection(self.currenttasks.index)]
        self.currentmanualtasks = self.currenttasks.loc[self.allmanualtasks.index.intersection(self.currenttasks.index)]
        self.currentautomatabletasks = self.currenttasks.loc[self.allautomatabletasks.index.intersection(self.currenttasks.index)]
        self.numcurrautomatedtasks = len(self.currentautomatedtasks.index)
        self.numcurrmanualtasks = len(self.currentmanualtasks.index)

    def hire(self):
        goalemployees = round(self.numcurrmanualtasks * self.expectedoutputq)
        goalrobots = round(self.numcurrautomatedtasks * self.expectedoutputq)
        self.employees = goalemployees
        self.robots = goalrobots

    def produce(self):
        if self.numcurrautomatedtasks == 0:
            self.minautomated = -1
        else:
            self.minautomated = self.robots // self.numcurrautomatedtasks
        if self.numcurrmanualtasks == 0:
            self.minmanual = -1
        else:
            self.minmanual = self.employees // self.numcurrmanualtasks
        if self.minautomated == -1:
            self.outputq = self.minmanual
        elif self.minmanual == -1:
            self.outputq = self.minautomated
        else:
            self.outputq = min([self.minautomated,self.minmanual])
        self.outputaddedval = np.mean(self.currenttasks['ADDEDVAL'])

    def budget(self):
        self.revenues = self.outputq * self.outputaddedval * self.model.pricefactor
        self.salarycosts = self.employees * self.model.salary
        self.robotcosts = self.robots * self.model.robotunitvalue * (self.model.intrate + self.model.deprate)
        ebrd = self.revenues - self.salarycosts - self.robotcosts
        if ebrd > 0:
            self.rdexpenses = ebrd * self.model.rdintensity
        else:
            self.rdexpenses = 0
        self.rdstock += self.rdexpenses
        self.opnetincome = self.revenues - self.salarycosts - self.robotcosts - self.rdexpenses
        self.expectedoutputq = self.outputq * (1 + self.model.expectedgrowth)
        self.opnetincomehist.append(self.opnetincome)
        if self.model.stepcounter >= 3 and self.opnetincomehist[self.model.stepcounter] < 0:
            if self.opnetincomehist[self.model.stepcounter -1] < 0 and self.opnetincomehist[self.model.stepcounter -2] < 0:
                self.model.toremove.append(self)

    def negotiatesalaries(self):
        if self.numcurrmanualtasks > 0:
            self.expectedemployeeproductivity = ((self.outputaddedval * self.model.pricefactor) -
                                                 (self.numcurrautomatedtasks * self.model.robotunitvalue *
                                                  (self.model.intrate + self.model.deprate)))
            self.numcurrmanualtasks
        else:
            self.expectedemployeeproductivity = 0
        self.expectedemployees = round(self.numcurrmanualtasks * self.expectedoutputq)

    def innovate(self):
        chance = np.random.poisson(self.rdstock / self.model.averagerdstock, 1)
        self.productinnovation = 0
        self.processinnovation = 0
        self.everythingautomated = 0
        self.belowthresholdaddedval = 0
        self.attemptprocessinn = 0
        if len(self.currentautomatabletasks.index) == 0:
            self.everythingautomated = 1
        if self.outputaddedval < self.model.thresholdoutputaddedval:
            self.belowthresholdaddedval = 1
        if len(self.currentautomatabletasks.index) > 0 and self.outputaddedval >= \
                self.model.thresholdoutputaddedval:
            self.attemptprocessinn = 1
        if chance > 1:
            if len(self.currentautomatabletasks.index) > 0 and self.outputaddedval >= \
                    self.model.thresholdoutputaddedval:
                tasktoinnovate = self.allautomatabletasks['KNOWLEDGE'].idxmax()
                self.knowntasks.set_value(tasktoinnovate,'KNOWLEDGE',
                                          self.knowntasks.loc[tasktoinnovate,'KNOWLEDGE'] + 0.9)
                if self.knowntasks.loc[tasktoinnovate,'KNOWLEDGE'] > 0.90:
                    self.knowntasks.set_value(tasktoinnovate,'KNOWLEDGE', 0.90)
                    self.processinnovation = 1
                else:
                    higheraddedvaltasks = self.model.taskset[self.model.taskset['ADDEDVAL'] > self.outputaddedval]
                    unknownhigheraddedvaltasks = \
                        higheraddedvaltasks[higheraddedvaltasks.index.isin(self.knowntasks.index) == False]
                    if len(unknownhigheraddedvaltasks.index) == 0:
                        self.model.generatetasks()
                        higheraddedvaltasks = self.model.taskset[self.model.taskset['ADDEDVAL'] >
                                                         self.outputaddedval]
                        unknownhigheraddedvaltasks = \
                            higheraddedvaltasks[higheraddedvaltasks.index.isin(self.knowntasks.index) == False]
                    tasktodiscover = np.random.choice(unknownhigheraddedvaltasks.index, 1)
                    self.knowntasks = self.knowntasks.append(unknownhigheraddedvaltasks.loc[tasktodiscover])
                    self.knowntasks.set_value(tasktodiscover, 'KNOWLEDGE', 0)
                    self.productinnovation = 1

    def step(self):
        self.settasks()
        self.hire()
        self.produce()
        self.budget()
        self.negotiatesalaries()
        if self.model.stepcounter >=1:
            self.innovate()


class InnEmpModel(Model):
    def __init__(self, nfirms, ntasks, nperformedtasks, numberknowntasks, probautomated, expectedoutputq, rdintensity,
                 productinnovationpropensity, expectedgrowth):
        self.running = True
        self.nfirms=nfirms
        self.ntasks=ntasks
        self.nperformedtasks=nperformedtasks
        self.rdintensity=rdintensity
        self.expectedgrowth=expectedgrowth
        self.productinnovationpropensity=productinnovationpropensity
        rdstock = 1
        self.activepopulation = self.nfirms * expectedoutputq * self.nperformedtasks
        self.schedule = RandomActivation(self)
        self.firmlist = []
        self.emplist = []
        self.intrate = 0.05
        self.deprate = 0.2
        self.salary = 1
        self.robotunitvalue = 1
        self.pricefactor = 1
        self.averagerev = 0
        self.averageoutputaddedval = 0
        self.averageproductivity = 0
        self.averagerdstock = 0
        self.stepcounter = 0
        self.taskgeneration = 0
        self.productinnovationratehist=[]
        self.processinnovationratehist=[]
        self.everythingautomatedratehist = []
        self.belowthresholdaddedvalratehist = []
        self.attemptprocessinnratehist = []
        self.firmreplacementratehist=[]
        self.taskset= pd.DataFrame()
        self.toremove = []
        self.generatetasks()
        for i in range(nfirms):
            afirm = Firm(i, self, numberknowntasks, probautomated, expectedoutputq, rdstock)
            self.schedule.add(afirm)
            self.firmlist.append(afirm)
        self.datacollector = DataCollector(model_reporters={"averageoutputaddedval": lambda m:
        m.averageoutputaddedval, "salary": lambda m: m.salary, "averagerdstock": lambda m:
        m.averagerdstock, "productinnovationrate": lambda m: m.productinnovationrate,
                                                            "processinnovationrate": lambda m: m.processinnovationrate,
                                                            "employmentrate": lambda m:
                                                            m.employmentrate, "wageshare": lambda m: m.wageshare,
                                                            "averageproductivity": lambda m:
                                                            m.averageproductivity,
                                                            "firmreplacementrate": lambda m: m.firmreplacementrate,
                                                            "weightedaverageprofitmargin": lambda
                                                                m: m.weightedaverageprofitmargin}, agent_reporters={})

    def step(self):
        self.schedule.step()
        self.outputmarketdynamics()
        self.labormarketdynamics()
        self.reportoutputmarketstats()
        self.reportinnovationstats()
        self.reportlabormarketstats()
        self.datacollector.collect(self)
        self.recycle()
        self.pricefactor = self.nextsteppricefactor
        self.salary = self.nextstepsalary
        self.stepcounter += 1

    def generatetasks(self):
        activitiessample = np.random.choice(7, self.ntasks, p=[0.07, 0.14, 0.16, 0.12, 0.17, 0.16, 0.18])

        addedvalssample = np.random.choice(range(1 + (self.taskgeneration * 10), 11 + (self.taskgeneration * 10)),
                                           self.ntasks)
        skillreqprobs = [0.09, 0.18, 0.2, 0.25, 0.64, 0.69, 0.78]
        skillreqsample = []
        for k in activitiessample:
            skillreq = np.random.choice((0.90, 1), 1, p=[skillreqprobs[k], 1 - skillreqprobs[k]])[0]
            skillreqsample.append(skillreq)
        self.taskset = pd.concat([self.taskset, pd.DataFrame({'ACTIVITY': activitiessample,
                                                              'ADDEDVAL': addedvalssample,
                                                              'SKILLREQ': skillreqsample})], ignore_index=True)
        self.taskgeneration += 1

    def recycle(self):
        lastid = self.firmlist[-1].unique_id

        for afirm in self.toremove:
            afirm.active = False
            self.schedule.remove(afirm)
            self.firmlist.remove(afirm)
        if len(self.toremove) > 0:
            totalnumberknowntasks = 0
            totalnumberautomatedtasks = 0
            totaloutputq = 0
            totalrdstock = 0
            for afirm in self.firmlist:
                totalnumberknowntasks += len(afirm.knowntasks)
                totalnumberautomatedtasks += len(afirm.allautomatedtasks)
                totaloutputq += afirm.outputq
                totalrdstock += afirm.rdstock
            averagenumberknowntasks = round(totalnumberknowntasks / len(self.firmlist))
            averagenumberautomatedtasks = totalnumberautomatedtasks / len(self.firmlist)
            averageprobautomated = averagenumberautomatedtasks / averagenumberknowntasks
            averageoutputq = round(totaloutputq / len(self.firmlist))
            averagerdstock = totalrdstock / len(self.firmlist)
        for afirm in self.toremove:
            newfirm = Firm(lastid + 1, self, averagenumberknowntasks, averageprobautomated, averageoutputq
                           * (1 + self.expectedgrowth), averagerdstock)
            for step in range(self.stepcounter + 1):
                newfirm.opnetincomehist.append(0)
            self.schedule.add(newfirm)
            self.firmlist.append(newfirm)
            lastid += 1
        self.toremove = []

    def reportoutputmarketstats(self):
        self.averagerev = self.totalrev / self.nfirms
        previousaverageoutputaddedval = self.averageoutputaddedval
        self.averageoutputaddedval = self.totaloutputaddedval / self.totaloutputq
        self.outputaddedvalpercapita = self.totaloutputaddedval / self.activepopulation
        self.outputaddedvalperemployee = self.totaloutputaddedval / self.totalemployed
        self.weightedaverageprofitmargin = self.totalopnetincome / self.totalrev
        self.firmreplacementrate = len(self.toremove) / self.nfirms
        self.firmreplacementratehist.append(self.firmreplacementrate)
        self.firmreplacementrateovertime = np.mean(self.firmreplacementratehist)
        if previousaverageoutputaddedval != 0:
            self.outputaddedvalincreaserate = self.averageoutputaddedval / previousaverageoutputaddedval - 1
        else:
            self.outputaddedvalincreaserate = 0

    def outputmarketdynamics(self):
        self.totaloutputq = 0
        self.totaloutputaddedval = 0
        self.totalrev = 0
        self.totalopnetincome = 0
        for afirm in self.firmlist:
            self.totaloutputq += afirm.outputq
            self.totaloutputaddedval += afirm.outputq * afirm.outputaddedval
            self.totalrev += afirm.revenues
            self.totalopnetincome += afirm.opnetincome
        self.nextsteppricefactor = self.totalrev / self.totaloutputaddedval

    def reportinnovationstats(self):
        self.totalrdstock = 0
        self.productinnovations = 0
        self.processinnovations = 0
        self.totaleverythingautomated = 0
        self.totalbelowthresholdaddedval = 0
        self.totalattemptprocessinn = 0
        self.totalnumcurrmanualtasks = 0
        self.totalnumcurrautomatedtasks = 0
        outputvallist = []
        for afirm in self.firmlist:
            self.totalrdstock += afirm.rdstock
            self.productinnovations += afirm.productinnovation
            self.processinnovations += afirm.processinnovation
            self.totaleverythingautomated += afirm.everythingautomated
            self.totalbelowthresholdaddedval += afirm.belowthresholdaddedval
            self.totalattemptprocessinn += afirm.attemptprocessinn
            self.totalnumcurrmanualtasks += afirm.numcurrmanualtasks
            self.totalnumcurrautomatedtasks += afirm.numcurrautomatedtasks
            outputvallist.append(afirm.outputaddedval)
        self.everythingautomatedrate = self.totaleverythingautomated / self.nfirms
        self.belowthresholdaddedvalrate = self.totalbelowthresholdaddedval / self.nfirms
        self.attemptprocessinnrate = self.totalattemptprocessinn / self.nfirms
        self.averagerdstock = self.totalrdstock / self.nfirms
        self.productinnovationrate = self.productinnovations / self.nfirms
        self.productinnovationratehist.append(self.productinnovationrate)
        self.processinnovationrate = self.processinnovations / self.nfirms
        self.processinnovationratehist.append(self.processinnovationrate)
        self.processinnovationrateovertime = np.mean(self.processinnovationratehist)
        self.productinnovationrateovertime = np.mean(self.productinnovationratehist)
        self.everythingautomatedratehist.append(self.everythingautomatedrate)
        self.belowthresholdaddedvalratehist.append(self.belowthresholdaddedvalrate)
        self.attemptprocessinnratehist.append(self.attemptprocessinnrate)
        self.everythingautomatedrateovertime = np.mean(self.everythingautomatedratehist)
        self.belowthresholdaddedvalrateovertime = np.mean(self.belowthresholdaddedvalratehist)
        self.attemptprocessinnrateovertime = np.mean(self.attemptprocessinnratehist)
        self.currmanualtasksrate = self.totalnumcurrmanualtasks / (self.totalnumcurrmanualtasks +
                                                                   self.totalnumcurrautomatedtasks)
        self.currautomatedtasksrate = self.totalnumcurrautomatedtasks / (self.totalnumcurrmanualtasks +
                                                                         self.totalnumcurrautomatedtasks)
        self.thresholdoutputaddedval = np.percentile(outputvallist, self.productinnovationpropensity)

    def reportlabormarketstats(self):
        self.wageshare = self.totalsalaries / self.totalrev
        self.unemploymentrate = 1 - self.employmentrate
        self.humanshareofinput = self.totalemployed / (self.totaloutputq * self.nperformedtasks)
        self.outputaddedvalperemployee = self.totaloutputaddedval / self.totalemployed
        self.salarygrowthrate = self.nextstepsalary / self.salary - 1

    def labormarketdynamics(self):
        totalexpectedproductivity = 0

        totalexpectedemployees = 0
        previousaverageproductivity = self.averageproductivity
        self.totalemployed = 0
        self.activepopulation = self.totaloutputq * self.nperformedtasks
        for afirm in self.firmlist:
            totalexpectedproductivity += afirm.expectedemployeeproductivity * afirm.expectedemployees
            totalexpectedemployees += afirm.expectedemployees
            self.totalemployed += afirm.employees
        self.totalsalaries = self.salary * self.totalemployed
        self.employmentrate = (self.totalemployed / self.activepopulation)
        employeebargainingpower = self.employmentrate
        self.averageproductivity = totalexpectedproductivity / totalexpectedemployees
        if previousaverageproductivity != 0:
            self.productivitygrowthrate = self.averageproductivity / previousaverageproductivity - 1
        else:
            self.productivitygrowthrate = 0
        self.nextstepsalary = self.averageproductivity * employeebargainingpower
