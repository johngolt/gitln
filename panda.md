##### Computational tools

###### Statistical functions

Series and DataFrame have a method pct_change() to compute the percent change over a given number of periods
Series.cov() can be used to compute covariance between series (excluding missing values).
Analogously, DataFrame.cov() to compute pairwise covariances among the series in the DataFrame, also excluding NA/null values.
Correlation may be computed using the corr() method. Using the method parameter, several methods for computing correlations are provided.All of these are currently computed using pairwise complete observations. 
A related method corrwith() is implemented on DataFrame to compute the correlation between like-labeled Series contained in different DataFrame objects.
The rank() method produces a data ranking with ties being assigned the mean of the ranks (by default) for the group:
rank() is also a DataFrame method and can rank either the rows (axis=0) or the columns (axis=1). NaN values are excluded from the ranking.
rank optionally takes a parameter ascending which by default is true; when false, data is reverse-ranked, with larger values assigned a smaller rank.

rank supports different tie-breaking methods, specified with the method parameter:

average : average rank of tied group
min : lowest rank in the group
max : highest rank in the group
first : ranks assigned in the order they appear in the array

###### Window Functions

For working with data, a number of window functions are provided for computing common window or rolling statistics. Among these are count, sum, mean, median, correlation, variance, covariance, standard deviation, skewness, and kurtosis.

The rolling() and expanding() functions can be used directly from DataFrameGroupBy objects,
We work with rolling, expanding and exponentially weighted data through the corresponding objects, Rolling, Expanding and EWM.
Generally these methods all have the same interface. They all accept the following arguments:

window: size of moving window
min_periods: threshold of non-null data points to require (otherwise result is NA)
center: boolean, whether to set the labels at the center (default is False)
We can then call methods on these rolling objects. These return like-indexed objects:
They can also be applied to DataFrame objects. This is really just syntactic sugar for applying the moving window operator to all of the DataFrame’s columns:

##### Visualization

If the index consists of dates, it calls gcf().autofmt_xdate() to try to format the x-axis nicely as per above.
Plotting methods allow for a handful of plot styles other than the default line plot. These methods can be provided as the kind keyword argument to plot(), and include:

‘bar’ or ‘barh’ for bar plots
‘hist’ for histogram
‘box’ for boxplot
‘kde’ or ‘density’ for density plots
‘area’ for area plots
‘scatter’ for scatter plots
‘hexbin’ for hexagonal bin plots
‘pie’ for pie plots
In addition to these kind s, there are the DataFrame.hist(), and DataFrame.boxplot() methods, which use a separate interface.

Finally, there are several plotting functions in pandas.plotting that take a Series or DataFrame as an argument. These include:

Scatter Matrix
Andrews Curves
Parallel Coordinates
Lag Plot
Autocorrelation Plot
Bootstrap Plot
RadViz
DataFrame.hist() plots the histograms of the columns on multiple subplots:
The by keyword can be specified to plot grouped histograms

Boxplot can be drawn calling Series.plot.box() and DataFrame.plot.box(), or DataFrame.boxplot() to visualize the distribution of values within each column.
You can create a stratified boxplot using the by keyword argument to create groupings.

You can create area plots with Series.plot.area() and DataFrame.plot.area(). Area plots are stacked by default. To produce stacked area plot, each column must be either all positive or all negative values.

When input data contains NaN, it will be automatically filled by 0. If you want to drop or fill by different values, use dataframe.dropna() or dataframe.fillna() before calling plot.

Scatter plot can be drawn by using the DataFrame.plot.scatter() method. Scatter plot requires numeric columns for the x and y axes. These can be specified by the x and y keywords.

You can create a pie plot with DataFrame.plot.pie() or Series.plot.pie(). If your data includes any NaN, they will be automatically filled with 0. A ValueError will be raised if there are any negative values in your data.
For pie plots it’s best to use square figures, i.e. a figure aspect ratio 1. You can create the figure with equal width and height, or force the aspect ratio to be equal after plotting by calling ax.set_aspect('equal') on the returned axes object.

Note that pie plot with DataFrame requires that you either specify a target column by the y argument or subplots=True. When y is specified, pie plot of selected column will be drawn. If subplots=True is specified, pie plots for each column are drawn as subplots. A legend will be drawn in each pie plots by default; specify legend=False to hide it.
You can create density plots using the Series.plot.kde() and DataFrame.plot.kde() methods.

Lag plots are used to check if a data set or time series is random. Random data should not exhibit any structure in the lag plot. Non-random structure implies that the underlying data are not random. The lag argument may be passed, and when lag=1 the plot is essentially data[:-1] vs. data[1:].
Parallel coordinates is a plotting technique for plotting multivariate data, see the Wikipedia entry for an introduction. Parallel coordinates allows one to see clusters in data and to estimate other statistics visually. Using parallel coordinates points are represented as connected line segments. Each vertical line represents one attribute. One set of connected line segments represents one data point. Points that tend to cluster will appear closer together.
Andrews curves allow one to plot multivariate data as a large number of curves that are created using the attributes of samples as coefficients for Fourier series, see the Wikipedia entry for more information. By coloring these curves differently for each class it is possible to visualize data clustering. Curves belonging to samples of the same class will usually be closer together and form larger structures.
Autocorrelation plots are often used for checking randomness in time series. This is done by computing autocorrelations for data values at varying time lags. If time series is random, such autocorrelations should be near zero for any and all time-lag separations. If time series is non-random then one or more of the autocorrelations will be significantly non-zero. The horizontal lines displayed in the plot correspond to 95% and 99% confidence bands. The dashed line is 99% confidence band.
Bootstrap plots are used to visually assess the uncertainty of a statistic, such as mean, median, midrange, etc. A random subset of a specified size is selected from a data set, the statistic in question is computed for this subset and the process is repeated a specified number of times. Resulting plots and histograms are what constitutes the bootstrap plot.
RadViz is a way of visualizing multi-variate data. It is based on a simple spring tension minimization algorithm. Basically you set up a bunch of points in a plane. In our case they are equally spaced on a unit circle. Each point represents a single attribute. You then pretend that each sample in the data set is attached to each of these points by a spring, the stiffness of which is proportional to the numerical value of that attribute (they are normalized to unit interval). The point in the plane, where our sample settles to (where the forces acting on our sample are at an equilibrium) is where a dot representing our sample will be drawn. Depending on which class that sample belongs it will be colored differently. 

If you have more than one plot that needs to be suppressed, the use method in pandas.plotting.plot_params can be used in a with statement:

A value $x$ is a high-dimensional datapoint if it is an element of ${R} ^{d}$ We can represent high-dimensional data with a number for each of their dimensions, ${\displaystyle x=\left\{x_{1},x_{2},\ldots ,x_{d}\right\}} $. To visualize them, the Andrews plot defines a finite Fourier series:

${\displaystyle f_{x}(t)={\frac {x_{1}}{\sqrt {2}}}+x_{2}\sin(t)+x_{3}\cos(t)+x_{4}\sin(2t)+x_{5}\cos(2t)+\cdots }$ 
This function is then plotted for ${\displaystyle -\pi <t<\pi } -\pi <t<\pi$ . Thus each data point may be viewed as a line between$ {\displaystyle -\pi }$  and ${\displaystyle \pi }$ . This formula can be thought of as the projection of the data point onto the vector:

${\displaystyle \left({\frac {1}{\sqrt {2}}},\sin(t),\cos(t),\sin(2t),\cos(2t),\ldots \right)}$
If there is structure in the data, it may be visible in the Andrews' curves of the data.

##### Group By

By “group by” we are referring to a process involving one or more of the following steps:

Splitting the data into groups based on some criteria.
Applying a function to each group independently.
Combining the results into a data structure.
Out of these, the split step is the most straightforward. In fact, in many situations we may wish to split the data set into groups and do something with those groups. In the apply step, we might wish to do one of the following:

Aggregation: compute a summary statistic (or statistics) for each group. Some examples:

Compute group sums or means.
Compute group sizes / counts.
Transformation: perform some group-specific computations and return a like-indexed object. Some examples:

Standardize data (zscore) within a group.
Filling NAs within groups with a value derived from each group.
Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some examples:

Discard data that belongs to groups with only a few members.
Filter out data based on the group sum or mean.
Some combination of the above: GroupBy will examine the results of the apply step and try to return a sensibly combined result if it doesn’t fit into either of the above two categories.

###### Splitting

 pandas objects can be split on any of their axes. The abstract definition of grouping is to provide a mapping of labels to group names.  

The mapping can be specified many different ways:

A Python function, to be called on each of the axis labels.
A list or NumPy array of the same length as the selected axis.
A dict or Series, providing a label -> group name mapping.
For DataFrame objects, a string indicating a column to be used to group. Of course df.groupby('A') is just syntactic sugar for df.groupby(df['A']), but it makes life simpler.
For DataFrame objects, a string indicating an index level to be used to group.
A list of any of the above things.

pandas Index objects support duplicate values. If a non-unique index is used as the group key in a groupby operation, all values for the same index value will be considered to be in one group and thus the output of aggregation functions will only contain unique index values
Note that no splitting occurs until it’s needed. Creating the GroupBy object only verifies that you’ve passed a valid mapping.
By default the group keys are sorted during the groupby operation. You may however pass sort=False for potential speedups:
The groups attribute is a dict whose keys are the computed unique groups and corresponding values being the axis labels belonging to each group. In the above example we have:
With hierarchically-indexed data, it’s quite natural to group by one of the levels of the hierarchy.If the MultiIndex has names specified, these can be passed instead of the level number. 
Grouping with multiple levels is supported.
A DataFrame may be grouped by a combination of columns and index levels by specifying the column names as strings and the index levels as pd.Grouper objects.

Once you have created the GroupBy object from a DataFrame, you might want to do something different for each of the columns. Thus, using [] similar to getting a column from a DataFrame, you can do
A single group can be selected using get_group():

###### Aggregation

Once the GroupBy object has been created, several methods are available to perform a computation on the grouped data. 
An obvious one is aggregation via the aggregate() or equivalently agg() method:
Note that you could use the reset_index DataFrame function to achieve the same result as the column names are stored in the resulting MultiIndex:
The aggregating functions above will exclude NA values. Any function which reduces a Series to a scalar value is an aggregation function and will work, a trivial example is df.groupby('A').agg(lambda ser: 1).
With grouped Series you can also pass a list or dict of functions to do aggregation with, outputting a DataFrame:
On a grouped DataFrame, you can pass a list of functions to apply to each column, which produces an aggregated result with a hierarchical index
To support column-specific aggregation with control over the output column names, pandas accepts the special syntax in GroupBy.agg(), known as “named aggregation”, where

The keywords are the output column names
The values are tuples whose first element is the column to select and the second element is the aggregation to apply to that column. Pandas provides the pandas.NamedAgg namedtuple with the fields ['column', 'aggfunc'] to make it clearer what the arguments are. As usual, the aggregation can be a callable or a string alias.

```python
 animals.groupby("kind").agg(min_height=pd.NamedAgg(column='height', aggfunc='min'),max_height=pd.NamedAgg(column='height', aggfunc='max'),
  average_weight=pd.NamedAgg(column='weight', aggfunc=np.mean)
# pandas.NamedAgg is just a namedtuple. Plain tuples are allowed as well.
animals.groupby("kind").agg(min_height=('height', 'min'),max_height=('height', 'max'),average_weight=('weight', np.mean),
```

Additional keyword arguments are not passed through to the aggregation functions. Only pairs of (column, aggfunc) should be passed as **kwargs. If your aggregation functions requires additional arguments, partially apply them with functools.partial().

The transform method returns an object that is indexed the same (same size) as the one being grouped. The transform function must:

Return a result that is either the same size as the group chunk or broadcastable to the size of the group chunk (e.g., a scalar, grouped.transform(lambda x: x.iloc[-1])).
Operate column-by-column on the group chunk. The transform is applied to the first group chunk using chunk.apply.
Not perform in-place operations on the group chunk. Group chunks should be treated as immutable, and changes to a group chunk may produce unexpected results. For example, when using fillna, inplace must be False (grouped.transform(lambda x: x.fillna(inplace=False))).
(Optionally) operates on the entire group chunk. If this is supported, a fast path is used starting from the second chunk.
Transformation functions that have lower dimension outputs are broadcast to match the shape of the input array

Working with the resample, expanding or rolling operations on the groupby level used to require the application of helper functions. However, now it is possible to use resample(), expanding() and rolling() as methods on groupbys.
The filter method returns a subset of the original object.
The argument of filter must be a function that, applied to the group as a whole, returns True or False.
Alternatively, instead of dropping the offending groups, we can return a like-indexed objects where the groups that do not pass the filter are filled with NaNs.
Some operations on the grouped data might not fit into either the aggregate or transform categories. Or, you may simply want GroupBy to infer how to combine the results. For these, use the apply function, which can be substituted for both aggregate and transform in many standard use cases. However, apply can handle some exceptional use cases,
apply on a Series can operate on a returned value from the applied function, that is itself a series, and possibly upcast the result to a DataFrame: