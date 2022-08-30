#!/usr/bin/env python
# coding: utf-8

# In[11]:


#CATENATION AND USE OF THE \ IN BETWEEN CHARACTERS TO AVOID AN ERROR: string manipulation
boo= "precious my sweetheart"
baby= 'this isn\'t working'
bae= "ROFL. \U0001F923"
length= len(boo)
print(boo,length, baby,bae)


# In[ ]:





# In[97]:


# functions: to assighn a function you need to begin your statement with the 'def' statement 
def interest_rate(principal, rate, time):
    result= principal*(1 + rate)**time
    return result
interest_rate(1000, 20/100, 1)


# In[56]:


#exercise 1
print(3*3)
print(144/12)
print(1565%3)
print((22/7+3.14**2)*3/0.025)
print(int(144**0.5))
pi=22/7
print(pi**0.5)
print(round((((3/(11**0.5))**2)*0.5*(7/9))**-1,2))


# In[82]:


#exercise 2
name= "Tobias"
print((name+" ")* 500)


# In[76]:


#exercise 3
print(int(22/3))


# In[83]:


#exercise 4 and 5
print(name.lower())
print(name.upper())
len(name)


# In[86]:


#exercise 6
x, y, z= 2**5, 130/5, 6*7
print(y==z)
print((y>x) and (z>y))
print(x>(y or z))
print((2*x>3*y-5) or (-y<z-2*x))


# In[92]:


#EXERCISE 7
variable3= 1


# In[6]:


combine=(["Let's ", 'do ', 'this!'])
### START FUNCTION
def combine(words):
    # HINT: use the `.join()` string method to combine your words
    # type: `sentence = "".join(words)`
    sentence = "".join(words)
    return sentence
### END FUNCTION
    


# In[22]:



word_list = ['be','have','do','say','get','make','go','know','take','see','come','think',
     'look','want','give','use','find','tell','ask','work','seem','feel','leave','call']
check = "S"
res = [idx for idx in word_list if idx.upper().startswith(check.upper())]
#res = [idx for idx in word_list if idx[0].lower() == check.lower()]
print(res)


# In[ ]:





# In[29]:


my_list= [['ARON', '0.1'], ['BEY', '0.2'], ['ABI', '0.05'], ['ZBBY', '0.9'], ['KB', '0.4']]

result = []

for i in my_list: 
    if i[0][0] == 'a'.upper():
          result.append(i[0]) 

print(result)


# In[1]:


== ['xanadu', 'xyz', 'aardvark', 'apple', 'mix']
### START FUNCTION
def front_x(list_of_words):
   # your code here
   # HINT: create two lists (one will contain words that start with X, the other not):
   list_x = []
   list_not_x = [word for word in list_of_words if word[0].lower() != "x"]
   # HINT: loop through words 
   for word in list_of_words:
       # HINT: use `if` to check if `word[0] == "x"`
       if word[0] == "x":
       # HINT: add the word to the list it belongs using e.g. `list_x.append(word)`
           list_x.append(word)
       
   # HINT: sort the lists outside of the for loop using e.g. `list_x.sort()`
   list_x.sort()
   # HINT: add the lists together and return `list_x + list_not_x`        
   
   return list_x + sorted(list_not_x)
### END FUNCTION


# In[5]:




numbers = [2,3,4,5]

list(map(lambda a: a**2, numbers)) # We are mapping the lambda function to iterate over the list `numbers`


# In[6]:


3%2


# In[7]:


nums = [1, 2, 3, 4, 5, 6]

[x**2 for x in nums if x%2==0]


# In[17]:


nums = [1,1,1,2,2,3,3,3,5,5,1,1,1]
def remove_adjacent(nums):
   
    numstail = [i for i in range(0,len(nums))] 
    nums = nums + numstail

    for i in nums:
        if nums[i] == nums[i-1]:
            del nums[i]

    return nums[:-len(numstail)]


# In[15]:


import matplotlib.pyplot as plt
import numpy as np
x = np.arange(-5,5,0.1)     # range of values for z
mu = 0                      # mu = 0 for standard normal
sigma = 3                    # mu = 1 for standard normal

# now calculate f(x)
f = 1 / np.sqrt ((2 * np.pi * sigma ** 2)) * np.exp (-0.5 * ((x - mu) / sigma) ** 2)

plt.rcParams["figure.figsize"] = (10,6)
plt.plot(x,f,'k')
plt.show()


# In[29]:


import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd

x = np.arange(1.2,2.2,0.01)    # Range of values for height from 1.2m to 2.2m
mu = 1.7                       # mu -> distribution mean = 1.7m
sigma = 0.1                    # sigma -> standard deviation = 0.1m

# We now calculate f(x), the probability density function of our normal distribution. 
f = st.norm.pdf(x, loc = mu, scale = sigma)

p = st.norm.cdf(1.5, loc = mu, scale = sigma)
print(f'The probability that a random data scientist has a height of 1.5 m is {np.round(p*100,2)} %') # F(1.5), i.e. probability of observing a height <= 1.5m

# Plot the results
plt.rcParams["figure.figsize"] = (10,6)
plt.plot(x,f,'r', label = 'Height normal distribution')
plt.axvline(x = 1.5, color = 'b', linestyle = '--', label = '1.5m Tall')
plt.legend()
plt.show()


# In[30]:




from IPython.display import Image
from IPython.core.display import HTML
Image(url= "https://github.com/Explore-AI/Pictures/blob/master/PEP_8_Guide.jpg?raw=true")


# In[32]:


original= [1, 2, 2, 3]
newlist=[]

for item in original:
    if item in newlist:
        print ("You don't need to add "+str(item)+" again.")
    else:
        newlist.append(item)
        print ("Added "+str(item))

print (newlist)


# In[39]:


def remove_adjacent(nums):
    return[k for k, g in nums.groupby()]


# In[40]:


remove_adjacent([27, -5, 13, 13, 0]) == [27, -5, 13, 0]


# In[43]:





# In[44]:


str(9+1)


# In[45]:


str(0+1)


# In[46]:


1+2+3.0


# In[49]:


scores_1 = [2, 4, 3, 8]
scores_2 = [4, 4, 5, 3]
scores_1 + scores_2


# In[53]:


S=[2,3,4,5,7,8,32,2,13,4]
sorted(S)


# In[56]:


len(S)


# In[57]:


students = ["janneman", "jordan", "john", "jacob", "james"]

for i in range(students):
    if students[i][-1] != 'n' and students[i][-1] != 'b':
        print(students[i])


# In[65]:


a = [58, 45, 12]
a.extend([10])
print(a)


# In[90]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
get_ipython().run_line_magic('matplotlib', 'inline')

mu = 0
variance = 2
sigma = np.log(variance) ,

x = np.linspace(mu-3*variance,mu+3*variance, 100)

plt.plot(x, norm.pdf(x^3) , mu, sigma)


# In[2]:


# Contents of the module we are creating 
content = """
s = 'Hello ' 

def say_hi(name):
    print(s+name)

class Greet:
    pass
"""

# Write the above text to a file called my_module.py
# within our current working directory. 
with open('./my_module.py', 'w') as fp:
    fp.write(content)


# In[98]:


#Import the module we've just made! 
import my_module
#(2)
my_module.s
#(3)
my_module.say_hi('Nelson')


# In[4]:


import numpy as np
# Create an array by passing in a list of lists.
ratings = np.array([[94,89,63], [93,92,48], [92,94,56]])

ratings


# In[7]:


# Sum of each row.
ratings.sum(axis=0)


# In[176]:


#Import the module we've just made! 
import my_module
#(2)
my_module.s
#(3)
my_module.say_hi('Nelson')


# In[177]:


# Sum of each row.
ratings.sum(axis=1)


# In[178]:


# Min of each row.
ratings.min(axis=1)


# In[179]:


# Min of each row.
ratings.min(axis=0)


# In[180]:


import pandas as pd #dataframe


# In[288]:


#creating a list in dataframe using list of lists.note the double[[]], for numpy we create the dataframe by using np.array([[]]) 

# Create list of lists containing data.
list_df = [[32, 'Portugal', 94], [30, 'Argentina', 93], [25 , 'Brazil', 92]]

# Create index - names of players.
index = ['Christiano Ronaldo', 'Lionel Messi', 'Neymar']

# Create column names.
columns = ['Age', 'Nationality', 'Overall']

# Create dataframe by passing in data, index and columns.
pf=pd.DataFrame(data=list_df, index=index, columns=columns) #this makes the dataframe that we have created to be called pf


# In[289]:


# we can use the iloc or loc function to get or slice certain data from our dataframe.
#iloc[] uses index numbers whereas loc[] uses names
#the difference between this and slicing by columns is 1: it can take for both row and columns and 2, it disoalys everything inbetween the selected columns
pf.loc['Age':'Overall']


# In[290]:


pf[['Age', 'Nationality']]


# In[291]:


# Load data - pass 'Name' as our index column.
# For this exercise, we'll use football player data to evaluate our dataframe.
load_df = pd.read_csv('https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/fundamentals/football_players.csv', index_col='Name')

# Create dataframe called df.
df = pd.DataFrame(load_df)

# Use the head() function to look at the first 5 rows.
df.head()


# In[ ]:





# In[292]:


df[['Age', 'Preferred Positions']] # this is a form of slicing the dataframe called by columns. it selects just the mentioned columns thereby ignoring the columns between the specified columns. 


# In[293]:


''''By index and column

We can also select a subset of the dataframe using indices and columns in combination. Let's look at a few examples:'''

df[['Age', 'Nationality']].iloc[0:5]  #this selects the interested columns and slices the data by the rows in the iloc index


# In[294]:


df.iloc[0:5][['Age', 'Nationality']] #irrespective of the position the iloc when ccombined with the columns slicing assuming the row


# In[295]:


#Select rows 9-14 of the football player dataframe we've just been using
df.iloc[9:15]


# In[296]:


#Select, in order, only the "Preferred Positions", "Overall" and "Age" columns for the football player Neymar
df[["Preferred Positions", "Overall", "Age"]].loc['Neymar']


# In[297]:


#Select, in order, only the "Overall", "Age" and "Nationality" columns for all players with ages of 35 years or older
df[["Overall", "Age", "Nationality" ]][df["Age"]>= 35]


# In[298]:


# Sort by age from youngest to oldest (select first 5 entries).
df.sort_values('Age').head()


# In[299]:




# Sort by age from oldest to youngest (select first 5 entries).
df.sort_values('Age', ascending=False).head() #this sorts in descending order because the ascending statement is false 


# In[300]:




# Filter on players older than 30.
df[df['Age'] > 30]# the outer df is making it return a list if not we only get a true and false table for all the players in the dataframe


# In[301]:


#We can also pass multiple conditions in the square brackets by using the | and & operators. Note that each condition should be closed inside round brackets as well.
# Filter on players older than 30 and overall rating greater than 90.
df[(df['Age'] > 30) | (df['Overall'] > 90)] #or operator


# In[302]:


df[(df['Age'] > 30) & (df['Overall'] > 90)] #and operator


# In[303]:


#We can create new columns from existing ones by simply defining the new name as a string inside square brackets, 
#followed by the function or operation of the other column(s).


# Create column of rating per year of age.
df['Rating Per Year of Age'] = df['Overall'] / df['Age']

# Look at first 5 entries.
df['Rating Per Year of Age'].head()


# In[304]:


#
'''Deleting Columns

Columns can be deleted by using the drop() function. The arguments are the column name and the axis which should be equal to 1 if we are deleting columns and 0 if we are deleting rows.

# Drop column just created'''
df = df.drop('Rating Per Year of Age', axis=1)

df.head(15)


# In[305]:


'''We can also group the data according to desired criteria. Grouping in dataframes can be achieved using the groupby() function.

Depending on your application, you may need to call an aggregation function for the grouped data after the groupby() function call.
Examples of aggregation functions include mean(), sum(), min() and max(). This will result in a column of values from the chosen
aggregation (for numeric columns). Let's look at an example:'''


# Look at the average rating by age (first 5 rows).
df.groupby('Age').mean().head()


# In[306]:


#It is possible to group by more than one column. We simply need to pass the list of columns.

# Look at the average rating by age and nationality (first 15 rows).
df.groupby(['Age', 'Nationality']).mean().head(15)


# In[307]:



df.groupby('Age').mean().head()


# In[308]:


# Create function.
def position_type(s):

    """"This function converts the individual positions (abbreviations) and classifies it
    as either a forward, midfielder, back or goalkeeper"""

    if (s[-2] == 'T') | (s[-2] == 'W'):
        return 'Forward'
    elif s[-2] == 'M':
        return 'Midfielder'
    elif s[-2] == 'B':
        return 'Back'
    else:
        return 'GoalKeeper'

# Create position type column.
df['Preferred Positions Type'] = df['Preferred Positions'].apply(position_type)

# Look at first 5 entries.
df['Preferred Positions Type'].head()


# In[309]:


df


# In[310]:


df.groupby('Age').mean().head()


# In[311]:


df.info()


# In[323]:


df.dtypes #checking the types of our data and object means string


# In[446]:


cols = ['Overall', 'Acceleration', 'Aggression','Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve','Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving','GK handling', 'GK kicking', 'GK positioning', 'GK reflexes','Heading accuracy', 'Interceptions', 'Jumping', 'Long passing','Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions','Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed','Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']


# In[ ]:





# In[447]:


# Use applymap() function to transform all selected columns.
df[cols] = df[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# In[331]:


#List all columns for players that have 'CB' as at least one of their preferred positions.
df[df['Preferred Positions'].str.contains('CB')]


# In[340]:


#List the mean age of players of each nationality in descending order

df.groupby(['Nationality'])['Age'].mean().sort_values(ascending=False)# ascending = false makes it go in a descending order


# In[333]:


#Using the apply and lambda functions, convert the "Age" of all players older than 36 years into floats.


# In[334]:


df.dtypes #checking just hte data types


# In[335]:


df.info()#checking the data types


# In[38]:




# Import libraries 
import matplotlib.pyplot as plt #used for plotting data 
import numpy as np #used for mathematical operations
import pandas as pd #used to loading CSV data


# In[79]:


df = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/tips.csv")#pd.read_csv() its a fuction to get our files into our notebook


# In[40]:


df.head()


# In[41]:




title_day = df.groupby('day').sum() #group and sum data by the number of values for each ‘day’ category
print(title_day)


# In[42]:


df.dtypes


# In[43]:


week_day = title_day.total_bill.sort_values().index #sort the indices 
bill = title_day.total_bill.sort_values()

print(bill)


# In[44]:


print(week_day)


# In[45]:


# Plot total bill (y-axis) for day of the week (x-axis) 
# We only have to call a single line of code from matplotlib to produce the base graph. 
plt.bar(week_day, bill, color= 'orange')

# Set x and y axis titles
plt.ylabel('Total Bill')
plt.xlabel('\n Days of the Week(Thur-Sun)') # Note: '\n' creates a newline (try removing it and see what happens)  

# Set graph title
plt.title('Total bill of customers for Thur-Sun \n')

# Show graph
plt.show()


# In[46]:




title_time = df.groupby('time').sum() # Group and sum data by the number of values for each ‘time’ category
print(title_time)


# In[47]:


meal_time = title_time.tip.sort_values().index #Sort the indices 
tips = title_time.tip.sort_values()
print(tips)


# In[48]:


print(meal_time)


# In[49]:




# Plot a pie chart
# The `autopct` argument defines the format applied to the data labels. 
# The `startangle` argument determines which point in the pie to start plotting proportions from. 
# Full plot documentation can be found here: https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.pie.html 
plt.pie(tips, labels = meal_time, autopct='%1.1f%%',  startangle=140)

# Place the chart legend in a position which does not overlap with other components. 
plt.legend(loc="best")
plt.axis('equal')
plt.show()


# In[50]:




total_meals = {'31/01/1990': 1340, '28/02/1990': 1338, '31/03/1990': 1330, '30/04/1990': 1328, '31/05/1990': 1335, '30/06/1990': 1335}


# In[51]:


dates = list(total_meals.keys()) # Extract the dates (the dictionary keys of our data in this case)
x_ax = [date[3:5] for date in dates] # Extract the month from each date string
y_ax = list(total_meals.values()) # Extract the total number of meals consumed on each date as a Python list

# Plot the line graph
plt.plot(x_ax, y_ax, color='indigo')

# Set axis and graph titles
plt.xlabel('Month')
plt.ylabel('Number of Total meals sold')
plt.title('Line Graph Showing the Total Number of Meals Sold Over the First 6 Months of 1990 \n')

plt.show()


# In[52]:


print(dates)
print(y_ax)
print(x_ax)


# In[53]:


# For this plot, we need to access the underlying Axes object used to create our chart. 
# To display our data correctly, we also set the `figsize` argument to increase the size of the plot. 
fig, ax = plt.subplots(figsize=(10,5))

# Create the scatter plot, with the 'size' variable being coded as the marker colour. 
# We set the `alpha` parameter to make the markers slightly transparent to view overlapping points. 
scatter = ax.scatter(df['total_bill'], df['tip'], c=df['size'], alpha=0.5)#alpha= color intensity 

# We now create our legend based upon the underlying group size and colour assignments.
ax.legend(*scatter.legend_elements(), loc="best", title="Group Size")

# Set graph and axis titles
plt.title('Scatter Plot Showing the Average Amount Tipped vs Group Size \n')
plt.xlabel('Bill Total ($)')
plt.ylabel('Amount Tipped ($)')

plt.show()


# In[54]:


smokers=df.groupby('smoker').sum()


# In[77]:


x_axis= smokers.total_bill.sort_values().index
y_axis= smokers.total_bill.sort_values()

plt.bar(x_axis, y_axis, color= 'gold', alpha= 0.8)

plt.ylabel('Total Bill')
plt.xlabel ('\n Smoker Status')
plt.title('Smokers Total Bill\n')
plt.show()


# In[56]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

soccer = pd.read_csv("https://raw.githubusercontent.com/Explore-AI/Public-Data/master/Data/fundamentals/football_players.csv",low_memory=False)


# In[57]:




cols = ['Age', 'Overall', 'Acceleration', 'Aggression',
       'Agility', 'Balance', 'Ball control', 'Composure', 'Crossing', 'Curve',
       'Dribbling', 'Finishing', 'Free kick accuracy', 'GK diving',
       'GK handling', 'GK kicking', 'GK positioning', 'GK reflexes',
       'Heading accuracy', 'Interceptions', 'Jumping', 'Long passing',
       'Long shots', 'Marking', 'Penalties', 'Positioning', 'Reactions',
       'Short passing', 'Shot power', 'Sliding tackle', 'Sprint speed',
       'Stamina', 'Standing tackle', 'Strength', 'Vision', 'Volleys']

soccer[cols] = soccer[cols].apply(pd.to_numeric, errors='coerce', axis=1)


# In[58]:


soccer.head()


# In[ ]:





# In[59]:


soccer.describe()


# In[60]:


for column in soccer.describe().columns:
    soccer[column] = soccer[column].apply(lambda x: x if x<=100 else np.nan)

soccer.describe()


# In[61]:


soccer.iloc[-1,0]


# In[62]:


soccer.iloc[0,0]


# In[63]:




soccer.iloc[1:101,:] # selects rows 1 to 100 from the DataFrame


# In[64]:


soccer.iloc[0:11,0]


# In[65]:


soccer.sort_values('Agility').max()


# In[66]:




soccer = soccer.rename(columns = {"Preferred Positions": "Preferred_Positions"})
soccer.head()


# In[67]:




soccer.select_dtypes(include=['object']).head()


# In[68]:




soccer.select_dtypes(include=['float']).head()


# In[69]:


soccer.dtypes


# In[70]:


import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[71]:




fig, ax = plt.subplots(2,1, figsize=(10,8))

sns.histplot(soccer['Age'], bins=5, element="step", ax=ax[0])
sns.histplot(soccer['Age'], bins=5, element="step", kde=True, ax=ax[1])

ax[0].set_title('Histogram for Ages')
ax[1].set_title('Histogram of Ages with distribution')

plt.tight_layout()
plt.show()


# In[72]:



plt.hist(soccer['Age'], bins=5)
plt.show()


# In[73]:




fig, ax = plt.subplots(2,1, figsize=(10,8))

sns.histplot(soccer['Overall'], bins=5, element="step", ax=ax[0])
sns.histplot(soccer['Overall'], bins=5, element="step", kde=True, ax=ax[1])

ax[0].set_title('Histogram for Overall')
ax[1].set_title('Histogram of Overall with distribution')

plt.tight_layout()
plt.show()


# In[74]:



plt.hist(soccer['Overall'], bins= 5)


# In[75]:


sns.jointplot(data= soccer, x= 'Aggression', y='Age', kind= 'hex')

plt.suptitle('Jointplot showing the Relationship between Age and Aggression of Football Players', y=1.1)
plt.show()


# In[76]:




young_players = soccer[(soccer['Age'] > 22) & (soccer['Age'] < 27)] #filter ages

plt.figure(figsize=(8,5))#fuction for the visual size. 8 for height 5 for width

sns.boxplot(data=young_players, x='Age', y='Overall')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




