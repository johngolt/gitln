$$
NWRMSLE = \sqrt{ \frac{\sum_{i=1}^n w_i \left( \ln(\hat{y}_i + 1) - \ln(y_i +1) \right)^2 }{\sum_{i=1}^n w_i}} 
$$

其中$\hat{y}_i$is the predicted unit sales of an item and $y_i$ is the actual unit sales; `n` is the total number of rows in the test set. The weights, $w_i$, can be found in the `items.csv` file

For each`id` in the test set, you must predict the`unit_sales`. Because the metric uses `ln(y+1)`, submissions are validated to ensure there are no negative predictions.

| 数据集                | 描述                                                         |
| --------------------- | ------------------------------------------------------------ |
| `train.csv`           | includes the target `unit_sales` by `date`, `store_nbr`, and `item_nbr` and a unique `id` to label rows. The `onpromotion` column tells whether that `item_nbr` was on promotion for a specified `date` and `store_nbr`. |
| `test.csv`            | with the `date`, `store_nbr`, `item_nbr` combinations that are to be predicted, along with the `onpromotion` information. |
| `stores.csv`          | Store metadata, including `city`, `state`, `type`, and `cluster`. `cluster` is a grouping of similar stores. |
| `items.csv`           | Item metadata, including `family`, `class`, and `perishable`. |
| `transactions.csv`    | The count of sales transactions for each `date`, `store_nbr` combination. Only included for the training data timeframe. |
| `oil.csv`             | Daily oil price. Includes values during both the train *and* test data timeframe. |
| `holidays_events.csv` | Holidays and Events, with metadata                           |

