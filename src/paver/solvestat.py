'''PAVER solve statistics generation.'''

import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;
import matplotlib.lines;
import os;
import math;
import itertools;

import utils;

# disable warning on future change in return_type for boxplot()
import warnings;
warnings.simplefilter(action = "ignore", category = FutureWarning);

DEFAULTNUMPTS = 40;

def _calculateProfile(paver, data, minabscissa = None, maxabscissa = None, logabscissa = True) :
    '''Calculates a performance profile from given data.'''
    
    numpts = DEFAULTNUMPTS;
    if 'numpts' in paver.options :
        numpts = paver.options['numpts'];
            
    # set min/maxabscissa to min/max in data, if not given by user
    if minabscissa is None :
        minabscissa = data.min().min();
    if maxabscissa is None :
        maxabscissa = data.max().max();

    # for logarithmic scale, we better start not before or at 0.0
    if logabscissa and minabscissa <= 0.0 :
        raise BaseException("Minimal abscissa is below 0.0, but abscissa is setup for logscale, consider using minabscissa.");
    if np.isinf(maxabscissa) :
        #maxabscissa = 1000.0 * minabscissa;
        raise BaseException("Maximal abscissa is at infinity.");
    #maxabscissa = max(maxabscissa, 1.0);
    
    # setup ticks for profile
    if minabscissa == maxabscissa :
        raise BaseException("Minimal and maximal attribute values are equal.");
    else :
        # distribute numpts ticks in [minabscissa : maxabscissa + a bit]
        ticks = np.empty(numpts);
        for i in range(0, numpts) :
            if logabscissa :
                ticks[i] = math.exp(math.log(minabscissa) + i/(numpts-1.0) * (math.log(maxabscissa + 1e-6) - math.log(minabscissa)));
            else :
                ticks[i] = minabscissa + i/(numpts-1.0) * (maxabscissa-minabscissa);
    
    # compute profile data
    profile = pd.DataFrame(0, index = ticks, columns = data.columns);
    for c, r in data.iteritems() :
        # order ratios for solver c
        ordered = r.order();
        pos = 0;
        for t in ticks :
            # increase pos while ratio <= t
            while pos < len(ordered) and ordered[pos] <= t :
                pos += 1;
            profile[c][t] = pos;

    return profile;

    
def _barchart(data, datamax = None, bw = False) :
    '''Create a matplotlib barchart for given data.'''
    #data.drop(['virt. best', 'virt. worst']).plot(kind = 'bar');
    
    #means = outcome.stat[meantype + '. mean'];
    #devs = outcome.stat[meantype + '. std.'];
    
    # drop NaN's and infinities
    data = data[data.map(np.isfinite)];
    realdata = data;
    
    nsolvers = len(data.index);
    if 'virt. best' in data :
        realdata = realdata.drop('virt. best');
        nsolvers -= 1;
    if 'virt. worst' in data :
        realdata = realdata.drop('virt. worst');
        nsolvers -= 1;

    barstart = np.empty(nsolvers);
    ticks = np.empty(nsolvers);
    for s in range(nsolvers) :
        barstart[s] = s + 0.1;
        ticks[s] = s + 0.5;
        
    plt.clf(); # clear figure

    yerr = None;
    #if meantype == 'arith' :
    #    yerr = devs.drop(['virt. best', 'virt. worst']);
    rects = plt.bar(barstart, realdata, 0.8, yerr = yerr, color = 'k' if bw else 'b');
    
    if 'virt. best' in data :
        plt.plot([0, nsolvers+0.1], [data['virt. best'], data['virt. best']], label = 'virt. best', color = 'k' if bw else 'b');
    if 'virt. worst' in data :
        plt.plot([0, nsolvers+0.1], [data['virt. worst'], data['virt. worst']], label = 'virt. worst', color = 'k' if bw else 'g', linestyle = '--' if bw else '-');
    #if meantype == 'arith' :
    #    s = {-2 : devs['virt. best'],  -1 : devs['virt. worst']};
    #    for i in [-2, -1] :  # draw error bars for virt. best/worst by hand
    #        plt.plot([nsolvers+0.05 - 0.05 * (i+1), nsolvers+0.05 - 0.05 * (i+1)], [m[i]-s[i], m[i]+s[i]], color = 'r');
    #        plt.plot([nsolvers+0.03 - 0.05 * (i+1), nsolvers+0.07 - 0.05 * (i+1)], [m[i]-s[i], m[i]-s[i]], color = 'r');
    #        plt.plot([nsolvers+0.03 - 0.05 * (i+1), nsolvers+0.07 - 0.05 * (i+1)], [m[i]+s[i], m[i]+s[i]], color = 'r');

    plt.axes().set_xticks(ticks);
    plt.axes().set_xticklabels(realdata.index);
    plt.xlim(0, nsolvers+0.15);
    #if meantype == 'arith' :
    #    plt.ylim(0, 1.1 * (m[-1] + s[-1]));
    #else :
    ymin = min(0, 1.1 * data.min());
    ymax = 1.1 * data.max();
    if ymin == ymax :
        ymax += 1.0;
    plt.ylim(ymin, ymax);

    for s in range(nsolvers) :
        rect = rects[s];
        height = rect.get_height();
        if np.mod(data[s], 1) == 0 :
            text = str(data[s]);
        else :
            text = '{0:.2f}'.format(data[s]);
        if datamax is not None :
            if np.mod(datamax[s], 1) == 0 :
                text += ' / ' + str(datamax[s]);
            else :
                text += ' / {0:.2f}'.format(datamax[s]);
        plt.text(rect.get_x()+rect.get_width()/2., height + 0.01 * ymax, text, ha='center', va='bottom');
    #plt.ylabel('Percentage of instances');
    if ('virt. best' in data) or ('virt. worst' in data):
        plt.legend(ncol = 2, loc = 1, frameon = False);


def addCommandLineOptions(parser) :
    '''Make command line parser aware of command line options of solve statistics generator.'''

    parser.add_argument('--numpts', action = 'store', type = int, default = DEFAULTNUMPTS,
                        help = 'number of points in performance profiles (default: ' + str(DEFAULTNUMPTS) + ')');
    parser.add_argument('--novirt', action = 'store_true',
                        help = 'disables virtual solvers in all statistics');
    parser.add_argument('--refsolver', action = 'append', default = [],
                        help = 'reference solver (default: None)');
    parser.add_argument('--nosortsolver', action = 'store_true',
                        help = 'disable sorting of solvers by name');
    parser.add_argument('--bw', action = 'store_true',
                        help = 'do graphics in black & white');

class StatisticsGenerator():
    '''Computes and stores solve statistics based on the aggregated solving data and a given set of performance metrics.'''

    
    class Outcome() :
        '''Class to store a computed solve statistic.'''
        
        def __init__(self, metric, datafilter) :
            self.metric = metric;
            self.filter = datafilter;

            self.stat = None;
            self.data = None;
            
            self.refsolver = {};

            self.pprofilerel = None;
            self.pprofileabs = None;
            self.pprofileext = None;


    def __init__(self, metrics = list()) :
        '''Constructor.
        @param metrics A list of metrics to evaluate.
        '''
        
        self._metrics = metrics;
        self._results = None;

    def _calculateStatistics(self, paver, refsolvers, metric, df, f) :
        '''Calculates statistics for a particular filter on a particular data frame and stores corresponding outcome object.'''

        virt = True;
        if 'novirt' in paver.options :
            virt = not paver.options['novirt'];
        
        if f is None :
            # a simple filter that includes all instances
            f = pd.Series(True, index = df.index);
            f.name = "all instances";
        # print f.name, f.count();
        
        # apply filter and drop rows that are complete NaN
        fdf = df[f].dropna(how = 'all').convert_objects(convert_numeric=True);

        #if len(fdf.index) == 0 :
        #   print 'No data left to evaluate attribute', metric.attribute, 'after applying filter "' + f.name + '". Skipping statistics.';
        #   return
        
        # fill remaining NaN's with na value (only does something if f was a dataframe, too)
        if metric.navalue is not None :
            fdf = fdf.fillna(metric.navalue).convert_objects(convert_numeric=True);
        
        if virt :
            # add virtual best, worst (can depend on filter, so do it now, not earlier)
            if metric.multbydirection :
                normalized = fdf.mul(paver.instancedata['Direction'], axis = 0);
                
                fdfmin = normalized.min(axis = 1).mul(paver.instancedata['Direction']);
                fdfmax = normalized.max(axis = 1).mul(paver.instancedata['Direction']);
            else :
                fdfmin = fdf.min(axis = 1);
                fdfmax = fdf.max(axis = 1);
                
            if metric.betterisup :
                fdf['virt. best']  = fdfmax;
                fdf['virt. worst'] = fdfmin.reindex_axis(fdf.dropna(how='any').index);
            else :
                fdf['virt. best']  = fdfmin;
                fdf['virt. worst'] = fdfmax.reindex_axis(fdf.dropna(how='any').index);
            #print fdf.to_string();
        
        # create dataframe for statistics
        stat = pd.DataFrame(index = fdf.columns);
        
        # always count number of non-NaNs
        stat['count'] = fdf.count();
        
        # skip statistic if no instance left in filter
        if stat['count'].max() == 0 :
            return;
        
        # compute arithmetic, geometric, shifted geometric means
        if metric.means :
            stat['arith. mean'] = fdf.mean();
            stat['arith. std.'] = fdf.std();
            
            # compute geometric mean
            #===================================================================
            # Given a set of data points (v_1, ..., v_n), we compute
            #  (prod_i (v_i + shift)) ^ (1/n) - shift
            # for some shift >= -v_i.
            # shift = 0 is the usual geometric mean.
            # 
            # Internally, we compute the (sh.) geom mean as
            #  exp( (sum_i log(v_i+shift) ) / n ) - shift.
            #  
            # The geometric standard deviation is
            #  exp( sqrt( {sum_i log(v_i+shift)^2} / n  - log(mean + shift)^2 ) ).
            #===================================================================
            if (fdf.fillna(1.0) > 0.0).min().min() :
                fdflog = np.log(fdf);
                fdflogmean = fdflog.mean();
                stat['geom. mean'] = np.exp(fdflogmean);
                
                part1 = np.square(fdflog).mean();
                part2 = np.square(fdflogmean);
                
                #if part2 > part1 :
                #    if part2 > part1 + 1e-9 :
                #        stat['geom. std.'] = np.nan;
                #    return 0.0;
                stat['geom. std.'] = np.exp(np.sqrt(part1 - part2));
                stat['geom. std.'][part2 > part1].fillna(0.0);
                
            # compute shifted geometric mean, if shift is given
            if metric.shift > 0.0 and (fdf.fillna(metric.shift + 1) > -metric.shift).min().min():
                fdflog = np.log(fdf + metric.shift);
                fdflogmean = fdflog.mean();
                stat['sh.geom. mean'] = np.exp(fdflogmean) - metric.shift;

                part1 = np.square(fdflog).mean();
                part2 = np.square(fdflogmean);
                
                stat['sh.geom. std.'] = np.exp(np.sqrt(part1 - part2));
                stat['sh.geom. std.'][part2 > part1].fillna(0.0);

        # compute quantiles
        if 0.0 in metric.quantiles :
            stat['min'] = fdf.min();
        for q in metric.quantiles :
            if q not in [0.0, 1.0] :
                stat[str(q*100) + "%"] = fdf.quantile(q);
        if 1.0 in metric.quantiles :
            stat['max'] = fdf.max();

        haszero = (fdf.fillna(1.0) == 0.0).max();

        # count number of cases where close to reference solvers
        refsolveroutcomes = {};
        for refsolver in refsolvers :

            refsolstat = pd.DataFrame(index = fdf.columns);
            
            # skip statistics if there are 0's in the reference values
            ratiotoref = None;
            if (not haszero[refsolver]) and ('bool' not in fdf.get_dtype_counts()) :
                ratiotoref = fdf.div(fdf[refsolver], axis=0);
                #print ratiotoref.to_string();
            
                # just arithmetic mean for now
                if metric.means :
                    refsolstat['arith. mean'] = ratiotoref.mean();
                    refsolstat['arith. std.'] = ratiotoref.std();
                
                # compute quantiles
                if 0.0 in metric.quantiles :
                    refsolstat['min'] = ratiotoref.min();
                for q in metric.quantiles :
                    if q not in [0.0, 1.0] :
                        refsolstat[str(q*100) + "%"] = ratiotoref.quantile(q);
                if 1.0 in metric.quantiles :
                    refsolstat['max'] = ratiotoref.max();

            # check whether above/within/below reltol/abstol of reference solver
            worsethanref = None;
            closetoref = None;
            betterthanref = None;
            if metric.reltol is not None or metric.abstol is not None :
                worsethanref = pd.DataFrame(index = fdf.index, columns = fdf.columns);
                #closetoref = pd.DataFrame(index = fdf.index, columns = fdf.columns);
                betterthanref = pd.DataFrame(index = fdf.index, columns = fdf.columns);
                allreldiff = pd.DataFrame(index = fdf.index, columns = fdf.columns);
                allabsdiff = pd.DataFrame(index = fdf.index, columns = fdf.columns);
                for c, s in fdf.iteritems() :
                    if metric.betterisup :
                        reldiff = utils.relDiff(fdf[refsolver], s);
                        absdiff = fdf[refsolver] - s;
                    else :
                        reldiff = utils.relDiff(s, fdf[refsolver]);
                        absdiff = s - fdf[refsolver];
                        
                    if metric.multbydirection :
                        reldiff = reldiff.mul(paver.instancedata['Direction']);
                        absdiff = absdiff.mul(paver.instancedata['Direction']);

                    allreldiff[c] = reldiff;
                    allabsdiff[c] = absdiff;

                    # check where solver c is worse than reference solver, i.e., differences > tolerances 
                    cb = None;
                    if metric.reltol is not None :
                        cb = reldiff >= metric.reltol;
                    if metric.abstol is not None :
                        if cb is not None :
                            cb &= absdiff >= metric.abstol;
                        else :
                            cb = absdiff >= metric.abstol;
                    worsethanref[c] = cb;

                    # check where solver c is better than reference solver, i.e., differences < -tolerances 
                    cb = None;
                    if metric.reltol is not None :
                        cb = reldiff <= -metric.reltol;
                    if metric.abstol is not None :
                        if cb is not None :
                            cb &= absdiff <= -metric.abstol;
                        else :
                            cb = absdiff <= -metric.abstol;
                    betterthanref[c] = cb;

                # check where solver c is about the same as reference solver
                closetoref = ~betterthanref & ~worsethanref & ~np.isnan(allreldiff) & ~np.isnan(allabsdiff);
                
                # if we did not compute ratios, provide relative differences as data
                # (FIXME: a big ugly and maybe confusing?)
                if ratiotoref is None :
                    ratiotoref = allreldiff;
                
                #===============================================================
                # if metric.attribute == 'NumberOfNodes' and refsolver != 'virt. best' :
                #    print metric.attribute, refsolver;
                #    print allreldiff.to_string();
                #    print closetoref.to_string();
                #    print closetoref.sum();
                #    print betterthanref.to_string();
                #    print betterthanref.sum();
                #    print worsethanref.to_string();
                #    print worsethanref.sum();
                #    exit(0);
                #===============================================================
                
                # count how many are within reltol/abstol of reference solver
                refsolstat['better'] = betterthanref.sum();
                refsolstat['close'] = closetoref.sum();
                refsolstat['worse'] = worsethanref.sum();
            
            # store only if we computed something
            if len(refsolstat.columns) > 0 :
                if ratiotoref is not None :
                    refsolstat.insert(0, 'count', ratiotoref.count());
                else :
                    refsolstat.insert(0, 'count', closetoref.count());
                refsolveroutcomes[refsolver] = {'stat' : refsolstat,
                                                'data' : ratiotoref,
                                                'betterthanref' : betterthanref,
                                                'closetoref' : closetoref,
                                                'worsethanref' : worsethanref
                                                };

        # store results in Outcome object
        outcome = self.Outcome(metric, f);
        outcome.stat = stat;
        outcome.refsolver = refsolveroutcomes;
        outcome.data = fdf;

        # store results in _results
        if metric.category not in self._results :
            self._results[metric.category] = [];
        self._results[metric.category].append(outcome);

    def _calculateProfiles(self, paver, metric, df, f) :
        '''Calculates performance profiles for a particular metric and filter and stores corresponding outcome object.'''
        
        virt = True;
        if 'novirt' in paver.options :
            virt = not paver.options['novirt'];
    
        absprofile = None;
        relprofile = None;
        extprofile = None;
    
        if f is None :
            # a simple filter that includes all instances
            f = pd.Series(True, index = df.index);
            f.name = "all instances";
    
        # print f.name, f.count();
        
        # apply filter
        fdf = df[f].copy();
        
        if len(fdf.index) == 0 :
           print 'No data left to evaluate attribute', metric.attribute, 'after applying filter "' + f.name + '". Skipping profiles.';
           return
           
        if fdf.min().min() == fdf.max().max() :
           print 'Attribute', metric.attribute, 'is the same for all instances and solvers after applying filter "' + f.name + '". Skipping profiles.';
           return
    
        if metric.ppextended :
            # for each solver run, compute best among all other solvers (for each instance)
            extbest = {};
            for sr in fdf :
                fdfother = fdf.drop(sr, axis=1);
                if metric.multbydirection :
                    normalized = fdfother.mul(paver.instancedata['Direction'], axis = 0);
                    
                    if metric.betterisup :
                        extbest[sr] = normalized.max(axis = 1).mul(paver.instancedata['Direction']);
                    else :
                        extbest[sr] = normalized.min(axis = 1).mul(paver.instancedata['Direction']);
                else :
                    if metric.betterisup :
                        extbest[sr] = fdfother.max(axis = 1);
                    else :
                        extbest[sr] = fdfother.min(axis = 1);
            extbest = pd.DataFrame(extbest);
            #print fdf.to_string();
            #print extbest.to_string();
    
            # get ratio to best among others
            ratiotoextbest = fdf.div(extbest);
            
            # virtually best / worst should be the best / worst ratio
            if virt :
                ratiotoextbest['virt. best'] = ratiotoextbest.min(axis = 1);
                ratiotoextbest['virt. worst'] = ratiotoextbest.dropna(how='any').max(axis = 1);
                
            # compute profile
            try :
                extprofile = _calculateProfile(paver, ratiotoextbest, None, None, metric.pprelsemilog);
            except BaseException, e :
                print 'Could not generate extended performance profile for attribute', metric.attribute, 'filter', f.name;
                print e;
        
        # compute virtual best and worst
        if metric.multbydirection :
            normalized = fdf.mul(paver.instancedata['Direction'], axis = 0);
            
            fdfmin = normalized.min(axis = 1).mul(paver.instancedata['Direction']);
            fdfmax = normalized.max(axis = 1).mul(paver.instancedata['Direction']);
        else :
            fdfmin = fdf.min(axis = 1);
            fdfmax = fdf.max(axis = 1);
            
        if metric.betterisup :
            fdfvirtbest = fdfmax;
            fdfvirtworst = fdfmin;
        else :
            fdfvirtbest = fdfmin;
            fdfvirtworst = fdfmax;
    
        # include virt. best / worst solver into fdf
        if virt :
            fdf['virt. best'] = fdfvirtbest;
            fdf['virt. worst'] = fdfvirtworst.reindex_axis(fdf.dropna(how='any').index);
            #print fdf.to_string();
        
        # compute absolute profile
        if metric.ppabsolute :
            try :
                absprofile = _calculateProfile(paver, fdf, None, None, metric.ppabssemilog);
            except BaseException, e :
                print 'Could not generate absolute performance profile for attribute', metric.attribute, 'filter', f.name;
                print e;
    
        # compute relative profile
        if metric.pprelative :
            ratiotobest = fdf.div(fdfvirtbest, axis=0);
            try :
                relprofile = _calculateProfile(paver, ratiotobest, None, None, metric.pprelsemilog);
            except BaseException, e :
                print 'Could not generate relative performance profile for attribute', metric.attribute, 'filter', f.name;
                print e;
    
        # store results in Outcome object
        outcome = self.Outcome(metric, f);
        outcome.data = fdf;
        outcome.pprofilerel = relprofile;
        outcome.pprofileabs = absprofile;
        outcome.pprofileext = extprofile;
    
        # store results in _results
        if metric.category not in self._results :
            self._results[metric.category] = [];
        self._results[metric.category].append(outcome);

    def calculate(self, paver) :
        '''Calculates solve statistics according to given metrics for aggregated solve data and stores outcome in _results.'''
        virt = True;
        if 'novirt' in paver.options :
            virt = not paver.options['novirt'];

        refsolvers = [];
        if virt :
            refsolvers.append('virt. best');
        if 'refsolver' in paver.options :
            for solver in paver.aggrsolvedata.items :
                if solver in paver.options['refsolver'] :
                    refsolvers.append(solver);
                    
        sortcolumns = True;
        if 'nosortsolver' in paver.options :
           sortcolumns = not paver.options['nosortsolver'];
        
        self._results = {};
        for metric in self._metrics :
            
            assert paver.hasSolveAttribute(metric.attribute);
            
            df = paver.aggrsolvedata.loc[:, :, metric.attribute];

            # replace NaN's and values for failed instances by failvalue, if given
            if metric.failvalue is not None :
                # set all fails to NaN
                df = df.mask(paver.aggrsolvedata.loc[:, :, 'Fail'].astype(bool));
                df[paver.instancedata['Fail']] = np.nan;
                
                # fill all NaN's with failvalue
                df = df.fillna(metric.failvalue);

            # clip inside given interval
            assert metric.clip_lower <= metric.clip_upper;
            df = df.clip(metric.clip_lower, metric.clip_upper);

            if sortcolumns :
               # sort columns (solver names)
               df.sort(axis=1, inplace=True);

            # statistics for each filter
            for f in metric.filter :
                self._calculateStatistics(paver, refsolvers, metric, df, f);

            # performance profile for each ppfilter
            for f in metric.ppfilter :
                self._calculateProfiles(paver, metric, df, f);
                

    def getCategories(self) :
        '''Gets the metric categories of the computed solve statistics.''' 
        if self._results is None :
            return [];
        return self._results.keys();
    
    def writeText(self, category, outdir, fileprefix):
        '''Outputs statistics for a certain category in plain text form.'''
        
        out = open(os.path.join(outdir, fileprefix + '.txt'), 'w');
        
        count = 0;
        for outcome in self._results[category] :
            #print >> out, "<H4>";
            #print >> out, "<TABLE>";
            #print >> out, "<TD style='{width: 200pt}'>", outcome.metric.attribute, "</TD>";
            #print >> out, "<TD>Filter:", outcome.filter.name, "</TD>";
            #print >> out, "</TABLE>"
            #print >> out, "</H4>";
            print >> out, "Category:", category;
            print >> out, "Attribute:", outcome.metric.attribute;
            print >> out, "Filter:", outcome.filter.name;
            
            if outcome.stat is not None :
                # output data dataframe into separate file
                datafile = fileprefix + 'data{0:03d}'.format(count) + '.txt';
                outdata = open(os.path.join(outdir, datafile), 'w');
                print >> outdata, outcome.data.dropna(how='all').to_string(na_rep = "-",
                                                                           index_names = False);
                outdata.close();

                # write statistics
                print >> out, outcome.stat.transpose().to_string(na_rep = "-",
                                                                 index_names = False,
                                                                 float_format = "{0:.2f}".format);
                print >> out, "Data:", datafile;

            # write performance profiles
            if outcome.pprofilerel is not None :
                profilefile = fileprefix + 'profilerel{0:03d}'.format(count) + '.txt';
                outdata = open(os.path.join(outdir, profilefile), 'w');
                
                print >> outdata, 'Relative performance profile for', outcome.metric.attribute;
                print >> outdata, 'Filter:', outcome.filter.name;
                print >> outdata, outcome.pprofilerel.to_string();

                print >> out, "Relative performance profile:", profilefile;

            if outcome.pprofileabs is not None :
                profilefile = fileprefix + 'profileabs{0:03d}'.format(count) + '.txt';
                outdata = open(os.path.join(outdir, profilefile), 'w');
                
                print >> outdata, 'Absolute performance profile for', outcome.metric.attribute;
                print >> outdata, 'Filter:', outcome.filter.name;
                print >> outdata, outcome.pprofileabs.to_string();

                print >> out, "Absolute performance profile:", profilefile;
                
            if outcome.pprofileext is not None :
                profilefile = fileprefix + 'profileext{0:03d}'.format(count) + '.txt';
                outdata = open(os.path.join(outdir, profilefile), 'w');
                
                print >> outdata, 'Extended performance profile for', outcome.metric.attribute;
                print >> outdata, 'Filter:', outcome.filter.name;
                print >> outdata, outcome.pprofileext.to_string();

                print >> out, "Extended performance profile:", profilefile;

            # write comparison to reference solvers
            for refsolver, refsolveroutcome in outcome.refsolver.iteritems() :

                # output ratiodata into separate file
                data = None;
                if refsolveroutcome['data'] is not None :
                    data = refsolveroutcome['data'].drop(refsolver, axis=1).dropna(how='all');
                    if len(data.index) == 0 :
                        data = None;

                if data is not None :
                    # format ratios in .2f format
                    df = data.applymap(lambda x : '{0:.2f}'.format(x) if not np.isnan(x) else '-');
                    # mark instances where we found a solver to be better/close/worse than the reference solver
                    if refsolveroutcome['betterthanref'] is not None :
                        df = df.combine(refsolveroutcome['betterthanref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, r+'+', r));
                    if refsolveroutcome['closetoref'] is not None :
                        df = df.combine(refsolveroutcome['closetoref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, r+'=', r));
                    if refsolveroutcome['worsethanref'] is not None :
                        df = df.combine(refsolveroutcome['worsethanref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, r+'-', r));
                
                    datafile = fileprefix + 'ratio{0:03d}_'.format(count) + str(refsolver).translate(None, " ()',") + '.txt';
                    outdata = open(os.path.join(outdir, datafile), 'w');
                    print >> outdata, df.to_string();
                    outdata.close();
                
                stat = refsolveroutcome['stat'].transpose().drop(refsolver, axis=1);

                print >> out;
                print >> out, "Relative to", refsolver, ":";
                print >> out, stat.to_string(na_rep = "-",
                                             float_format = "{0:.2f}".format);
                if data is not None :
                    print >> out, 'Data:', datafile;

            count += 1;
            print >> out;
            
        out.close();

    def writeHTML(self, paver, category, outdir, fileprefix) :
        '''Outputs statistics for a certain category in HTML form.'''
        
        # whether to do black-and-white
        bw = 'bw' in paver.options and paver.options['bw'];
        if bw :
            plotstyle = [];
            for m in [''] + matplotlib.markers.MarkerStyle.markers.keys() :
                if not isinstance(m, basestring) and m != 'None' :
                    continue;
                for ls in matplotlib.lines.lineStyles.keys() :
                    if ls != 'None' and ls != ' ' :
                        plotstyle.append('k' + ls + str(m));
        else :
            plotstyle = '-o';

        out = open(os.path.join(outdir, fileprefix + '.html'), 'w');
        print >> out, "<HTML>";
        print >> out, "<HEAD>";
        print >> out, paver.htmlheader;
        print >> out, "<STYLE> .block {background:#f0f0f0; padding:10px; border: 1px solid #c2c2c2;} </STYLE>";
        print >> out, "</HEAD>";
        print >> out, "<BODY>";
        
        print >> out, "<H3>Statistics for category", category, "</H3>"
        
        # print table of contents
        count = 0;
        prevattr = '';
        print >> out, "<P><TABLE><TR><TH align=left>Attribute</TH><TH align=left> Filter</TH></TR>"
        for outcome in self._results[category] :
            print >> out, "<TR>";
            if outcome.metric.attribute != prevattr :
                print >> out, "<TD>", "<A href=#"+str(count)+">", outcome.metric.attribute, "</A>", '</TD>'
            else :
                print >> out, "<TD>&nbsp;</TD>";
            prevattr = outcome.metric.attribute;
            print >> out, "<TD>", "<A href=#"+str(count)+">", outcome.filter.name, "</A>", "</TD>"
            print >> out, "</TR>";
            count += 1;
        print >> out, "</TABLE></P>"

        print >> out, "<P>For a short description on the computed values, refer to <A href=documentation.html>the documentation</A>.</P>";
        
        count = 0;
        for outcome in self._results[category] :
            #print >> out, "<H4>";
            #print >> out, "<TABLE>";
            #print >> out, "<TD style='{width: 200pt}'>", outcome.metric.attribute, "</TD>";
            #print >> out, "<TD>Filter:", outcome.filter.name, "</TD>";
            #print >> out, "</TABLE>"
            #print >> out, "</H4>";
            print >> out, "<div class='block'>";
            print >> out, "<A name="+str(count)+">";
            print >> out, "<H4>", outcome.metric.attribute, "<BR>";
            print >> out, "Filter:", outcome.filter.name, "</H4>";
            print >> out, "</A>";
            
            print >> out, "<P><FONT SIZE=-1>";
            if not np.isneginf(outcome.metric.clip_lower) or not np.isposinf(outcome.metric.clip_upper) :
                print >> out, "Attribute values were projected onto interval [" + str(outcome.metric.clip_lower) + ', ' + str(outcome.metric.clip_upper) + '].<BR>';
            if outcome.metric.failvalue is not None :
                print >> out, "Missing values and values for failed instances substituted by", outcome.metric.failvalue, ".<BR>";
            if outcome.metric.navalue is not None :
                print >> out, "Missing values after filtering substituted by", outcome.metric.navalue, ".<BR>";
            print >> out, "</FONT></P>" 
            
            if outcome.stat is not None :
                # output data dataframe into separate file
                datafile = fileprefix + 'data{0:03d}'.format(count) + '.html';
                outdata = open(os.path.join(outdir, datafile), 'w');
                print >> outdata, "<HTML><HEAD>", paver.htmlheader, "</HEAD><BODY>";
                print >> outdata, outcome.data.dropna(how='all').to_html(na_rep = "-",
                                                                         index_names = False);
                print >> outdata, "</BODY>", "</HTML>";
                outdata.close();

                # write statistics
                table = outcome.stat.transpose().to_html(na_rep = "-",
                                                         index_names = False,
                                                         float_format = "{0:.2f}".format);
                table = table.replace("<th>minor</th>", "<th></th>", 1).replace("<th></th>", "<th align=left><A href=" + datafile + ">Data</A></th>", 1);
                print >> out, "<P>";
                print >> out, table;
                if 'sh.geom. mean' in outcome.stat :
                    print >> out, "<FONT SIZE=-1>geometric mean shift:", outcome.metric.shift, "</FONT>";
                print >> out, "</P>";
                
                print >> out, "<P>";
                # if we have only a count, do a plot for this one (probably that's why we have this metric)
                if (outcome.stat.columns == 'count').all() and len(outcome.refsolver) == 0 :
                    _barchart(outcome.stat['count'], bw = bw);
                    plt.title(outcome.metric.attribute + ' - ' + outcome.filter.name);

                    plotfile = fileprefix + 'count{0:03d}'.format(count);
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.png'));
                    if paver.options['figformat'] != 'png' :
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                    plt.close();
                    
                    print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";
                
                # bar charts for each type of mean
                if outcome.metric.means :
                    for meantype in ('arith', 'geom', 'sh.geom') :
                        if meantype + '. mean' not in outcome.stat :
                            continue;
                        
                        _barchart(outcome.stat[meantype + '. mean'], bw = bw);
                        plt.title(outcome.metric.attribute + ' - ' + meantype + '. means\nFilter: ' + outcome.filter.name);

                        plotfile = fileprefix + meantype + 'mean{0:03d}'.format(count);
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.png'));
                        if paver.options['figformat'] != 'png' :
                            with utils.Timer() :
                                plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                        plt.close();
                        
                        print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";
                
                # boxplot
                if outcome.metric.boxplot :
                    plt.clf();
                    if bw :
                        bp = outcome.data.boxplot(grid = False); # return_type = 'axes'
                        plt.setp(bp['boxes'], color='black')
                        plt.setp(bp['medians'], color='black')
                        plt.setp(bp['whiskers'], color='black')
                        plt.setp(bp['fliers'], color='black', marker='+')
                    else :
                        outcome.data.boxplot(); # return_type = 'axes'
                    plt.title(outcome.metric.attribute + '\nFilter: ' + outcome.filter.name);
                    
                    plotfile = fileprefix + 'boxplot{0:03d}'.format(count);
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.png'));
                    if paver.options['figformat'] != 'png' :
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                    plt.close();
                    
                    print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

                print >> out, "</P>";

            # plot performance profiles
            haveprof = outcome.pprofilerel is not None or outcome.pprofileabs is not None or outcome.pprofileext is not None;
            if haveprof :
                print >> out, "<P>";
            
            if outcome.pprofilerel is not None :
                plt.figure();
                outcome.pprofilerel.plot(logx = outcome.metric.pprelsemilog,
                                         style = plotstyle,
                                         title = "Relative performance profile (" + outcome.metric.attribute + ")");
                plt.xlabel(outcome.metric.attribute + ' at most this factor of best');
                plt.ylabel('Number of instances with ' + outcome.filter.name);
                plt.ylim(0, len(outcome.metric.ppfilter[0].index));
                
                plotfile = fileprefix + 'profilerel{0:03d}'.format(count);
                with utils.Timer() :
                    plt.savefig(os.path.join(outdir, plotfile + '.png'));
                if paver.options['figformat'] != 'png' :
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                plt.close();

                print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

            if outcome.pprofileabs is not None :
                plt.figure();
                outcome.pprofileabs.plot(logx = outcome.metric.ppabssemilog,
                                         style = plotstyle,
                                         title = "Absolute performance profile (" + outcome.metric.attribute + ")");
                plt.xlabel(outcome.metric.attribute);
                plt.ylabel('Number of instances with ' + outcome.filter.name);
                plt.ylim(0, len(outcome.metric.ppfilter[0].index));
                
                plotfile = fileprefix + 'profileabs{0:03d}'.format(count);
                with utils.Timer() :
                    plt.savefig(os.path.join(outdir, plotfile + '.png'));
                if paver.options['figformat'] != 'png' :
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                plt.close();

                print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

            if outcome.pprofileext is not None :
                plt.figure();
                outcome.pprofileext.plot(logx = outcome.metric.pprelsemilog,
                                         style = plotstyle,
                                         title = "Extended performance profile (" + outcome.metric.attribute + ")");
                plt.xlabel(outcome.metric.attribute + ' at most this factor of best excluding self');
                plt.ylabel('Number of instances with ' + outcome.filter.name);
                plt.ylim(0, len(outcome.metric.ppfilter[0].index));
                
                plotfile = fileprefix + 'profileext{0:03d}'.format(count);
                with utils.Timer() :
                    plt.savefig(os.path.join(outdir, plotfile + '.png'));
                if paver.options['figformat'] != 'png' :
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                plt.close();

                print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

            if haveprof :
                print >> out, "</P>";
                            
            for refsolver, refsolveroutcome in outcome.refsolver.iteritems() :

                # output ratiodata into separate file
                data = None;
                if refsolveroutcome['data'] is not None :
                    data = refsolveroutcome['data'].drop(refsolver, axis=1).dropna(how='all');
                    if len(data.index) == 0 :
                        data = None;

                if data is not None :
                    # format ratios in .2f format
                    df = data.applymap(lambda x : '{0:.2f}'.format(x) if not np.isnan(x) else '-');
                    # mark instances where we found a solver to be better/close/worse than the reference solver
                    if refsolveroutcome['betterthanref'] is not None :
                        df = df.combine(refsolveroutcome['betterthanref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, '<span style="{color: green}">'+r+'</span>', r));
                    if refsolveroutcome['closetoref'] is not None :
                        df = df.combine(refsolveroutcome['closetoref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, '<span style="{color: blue}">'+r+'</span>', r));
                    if refsolveroutcome['worsethanref'] is not None :
                        df = df.combine(refsolveroutcome['worsethanref'].drop(refsolver, axis=1),
                                        lambda r, c : np.where(c, '<span style="{color: red}">'+r+'</span>', r));
                
                    datafile = fileprefix + 'ratio{0:03d}_'.format(count) + str(refsolver).translate(None, " ()',") + '.html';
                    outdata = open(os.path.join(outdir, datafile), 'w');
                    print >> outdata, "<HTML><HEAD>", paver.htmlheader, "</HEAD><BODY>";
                    print >> outdata, df.to_html(index_names = False, escape = False);
                    print >> outdata, "</BODY>", "</HTML>";
                    outdata.close();
                
                stat = refsolveroutcome['stat'].transpose().drop(refsolver, axis=1);

                table = stat.to_html(na_rep = "-",
                                     index_names = False,
                                     float_format = "{0:.2f}".format);
                if data is not None :
                    table = table.replace("<th>minor</th>", "<th></th>", 1).replace("<th></th>", "<th align=left><A href=" + datafile + ">Data</A></th>", 1);
                
                print >> out, "<P>";
                print >> out, "Performance with respect to ", refsolver + ":", "<BR>";
                print >> out, "<FONT SIZE=-1>Tolerance:";
                if outcome.metric.reltol is not None :
                    print >> out, "relative", outcome.metric.reltol;
                if outcome.metric.abstol is not None :
                    print >> out, "absolute", outcome.metric.abstol;
                print >> out, "</FONT><BR>";
                print >> out, table;
                print >> out, "</P>";
                
                print >> out, "<P>";
                # boxplot
                if outcome.metric.boxplot and data is not None:
                    plt.clf();
                    if bw :
                        bp = data.boxplot(grid = False); # return_type = 'axes'
                        plt.setp(bp['boxes'], color='black')
                        plt.setp(bp['medians'], color='black')
                        plt.setp(bp['whiskers'], color='black')
                        plt.setp(bp['fliers'], color='black', marker='+')
                    else :
                        data.boxplot(); # return_type = 'axes'
                    plt.title(outcome.metric.attribute + ' w.r.t. ' + str(refsolver) + '\nFilter: ' + outcome.filter.name);
                    
                    plotfile = fileprefix + 'boxplot{0:03d}_'.format(count) + str(refsolver).translate(None, " ()',");
                    with utils.Timer() :
                        plt.savefig(os.path.join(outdir, plotfile + '.png'));
                    if paver.options['figformat'] != 'png' :
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                    plt.close();
                    
                    print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

                # bar charts for each type of mean
                if outcome.metric.means :
                    for meantype in ('arith', 'geom', 'sh.geom') :
                        if meantype + '. mean' not in refsolveroutcome['stat'] :
                            continue;
                        
                        _barchart(refsolveroutcome['stat'][meantype + '. mean'], bw = bw);
                        plt.title(outcome.metric.attribute + ' w.r.t. ' + str(refsolver) + ' - ' + meantype + '. means\nFilter: '+ outcome.filter.name);

                        plotfile = fileprefix + meantype + 'mean{0:03d}_'.format(count) + str(refsolver).translate(None, " ()',");
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.png'));
                        if paver.options['figformat'] != 'png' :
                            with utils.Timer() :
                                plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                        plt.close();
                        
                        print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

                # bar charts for number of instance close to/better than/worse than refsolver
                for a in ['close', 'better', 'worse'] :
                    if a in refsolveroutcome['stat'] :
                        _barchart(refsolveroutcome['stat'][a], refsolveroutcome['stat']['count'], bw = bw);
                        title = outcome.metric.attribute + ' ' + a + (' to ' if a == 'close' else ' than ') + str(refsolver) + ' (';
                        if outcome.metric.reltol is not None :
                            title += 'rel ' + '{0:.1f}%'.format(100.0*outcome.metric.reltol);
                            if outcome.metric.abstol is not None :
                                title += ', ';
                        if outcome.metric.abstol is not None :
                            title += 'abs ' + str(outcome.metric.abstol);
                        title += ')\nFilter: ' + outcome.filter.name;
                        plt.title(title);
    
                        plotfile = fileprefix + a + '{0:03d}_'.format(count) + str(refsolver).translate(None, " ()',");
                        with utils.Timer() :
                            plt.savefig(os.path.join(outdir, plotfile + '.png'));
                        if paver.options['figformat'] != 'png' :
                            with utils.Timer() :
                                plt.savefig(os.path.join(outdir, plotfile + '.' + paver.options['figformat']));
                        plt.close();
                        
                        print >> out, "<a href=" + plotfile + "." + paver.options['figformat'] + "><img src=" + plotfile + ".png width=300></a>";

                print >> out, "</P>";

            count += 1;
            print >> out, "</div><BR>";

        print >> out, "</BODY>", "</HTML>";
        out.close();
