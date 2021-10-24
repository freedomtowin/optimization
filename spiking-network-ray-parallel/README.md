# spiking-network

Since I am hitting my popular technology quota, I decided to also post about Spiking Networks. From my cloud-automated stock data collection project, I now have over a thousand rows of open-market hourly data. [You can see the processed data in the repo].

I've found that small market fluxuations are fairly easy to predict, but its much harder to predict the profitable price swings.

Input data: NVDA price differences\twitter\google news sentiment

Activation: Non-differentiable, spiking layer

Target: Next hour price difference

Objective: Minimize false positive and negatives

Validation time window: 1 month

Train accuracy: 0.61

Validation accuracy: 0.64

Optimization: KernelML

![spiking-network-results](https://github.com/freedomtowin/spiking-network/blob/main/spiking-network-results.PNG)