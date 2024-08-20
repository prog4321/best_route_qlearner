### APPLYING Q-LEARNING TO FIND THE FASTEST ROUTES IN SINGAPORE'S RAIL NETWORK (MRT)

Using Q-learning to find the fastest routes in Singapore's rail network (MRT):
https://prog4321.github.io/best_route_qlearner/

REFERENCES:
1. http://firsttimeprogrammer.blogspot.com/2016/09/getting-ai-smarter-with-q-learning.html
2. https://amunategui.github.io/reinforcement-learning/index.html
3. 'Artificial Intelligence Programming with Python' by Perry Xiao (Wiley)

Note: Please first read through Ref #1 above. Refs #2 and #3 both employ Ref #1's
concepts and Python code quite heavily.

Most examples on Q-learning that I have come across online are either very closely based
on Ref #1, or apply Q-learning in a simulated/gaming environment (for e.g. OpenAI's
Gym/Gymnasium environments).

I wanted to see if I could apply Q-learning in a real-life environment. I have chosen
to use Singapore's fairly extensive rail network (known as the MRT, or Mass Rapid Transit)
as the environment for this project. I wanted to see if I could build on Ref #1's approach
to find the best (shortest) route from a start node to an end node, but now factoring in
the different travelling times between train stations on the MRT - the route that
traverses the least number of stations may not necessarily be the route that is the
shortest in terms of travelling duration.

Here, I treat the travelling time between 2 train stations as a cost incurred. The longer
the travelling time, the higher the cost. I normalise this cost through dividing it by the
maximum cost, which I define as the maximum of the travelling times between any 2 stations,
plus a chosen waiting time (i.e. wait cost) for the period (peak hour or off-peak). I then
use 1 to deduct this normalised cost, in order to get a reward value. So the lesser the cost,
the greater the reward. I further multiply this reward value with what I call a
reward coefficient. After running a grid search-like algorithm
(pls see https://www.kaggle.com/code/prog4321/using-q-learning-to-find-the-fastest-train-routes)
to find the best performing values for the alpha and gamma hyperparameters, there is also a
range of values for this reward coefficient that work well for the chosen alpha and gamma values.
I have put in a set of default values for initialising the BestRouteQLearner object that will
work well with the Singapore MRT dataset - these default values may need to be adjusted
accordingly for optimal usage on other datasets.

The core MRT data can be found in the 'mrt_data.xlsx' file inside the Data folder of this
project. It includes 3 worksheets:

1. Nodes
This worksheet stores the Station Codes and Station Names. Each Station Code has to be unique
(just like in a table in a relational database). NB: The train interchanges must share the same
Station Name (for e.g. 'Bishan' for station codes NS17 and CC15). As the program evaluates the
routes and detects a repeated instance of the Station Name, it will factor in a wait cost.
For e.g. the commuter is on a route that makes a transit at Bishan interchange, going from the
NS17 Bishan train platform to the CC15 Bishan train platform. The cost value indicated in the
Routes worksheet (detailed shortly) will indicate the time needed to *walk* from the NS17
platform to the CC15 platform. If the commuter is to board the train at the CC15 platform as
part of the route, naturally there is also a waiting time (wait cost) of having to wait for the
next train at the CC15 platform. This wait cost will be the chosen wait cost for the peak hour
and off-peak periods, and can be set when the BestRouteQLearner object is initialised.

2. Routes
Here, we indicate the Start Station Codes and the End Station Codes, as well as the travelling
time (i.e. cost) between the two respective stations. In the case of interchanges, the cost
to be indicated here will be the walking time between the 2 platforms of that interchange.
(The Station Names in this workwheet simply use the vlookup function in Excel to reflect the
Station Names that were entered in the Nodes worksheet.)

3. Interchanges
Here, we list down the Station Codes that belong to each interchange. The program will
check the route for any transits at the interchanges (as mentioned above) and add a
wait cost to the route if applicable.

These 3 worksheets are then saved as CSV files in the same Data folder. To ensure
data integrity, it would naturally be best to have the data stored in a well-designed
relational database, but for simplicity here I just use CSV files. (Furthermore, the
data for a rail system does not change very often.) There will be code run in the 
main.py file to extract the data from these 3 CSV files. Specifically, for the 'fit'
method of the BestRouteQLearner object, we require numpy arrays in the following 
format:

1. Nodes
Node ID, Node Name
Essentially the Station Code and Station Name

2. Routes
Start Node ID, End Node ID, Cost
I.e. the start Station Code, the end Station Code, and the travelling/waiting time
between the 2 stations. NB: the program assumes the time taken to go from
Station A to Station B will be the same as the time taken to go from Station B to
Station A.

3. Interchanges
All the Node IDs for each interchange are listed in subsequent columns,
starting at the 2nd column, as in the Interchanges worksheet.

Amend the code in main.py accordingly to cater for different input formats, as
long as it is able to extract the required numpy arrays in the format mentioned above.

For the front-end, for simplicity of demonstation I have chosen to use Pyscript,
along with some simple Javascript for functionality and CSS for formatting.

Please include a credit under my Github moniker 'prog4321' if you wish to use/modify
the code for your own projects. Thank you.

I will try to include more documentation for this project where and when I can.
