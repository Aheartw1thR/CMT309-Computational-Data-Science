# This is a template of the submission file
import numpy as np
import string
import pandas as pd
import re


# Question 1
def all_rounder(*args):
    '''
    performs passed sequence method of a given sequence
    :param args: arguments passed in
    :return: the result
    '''
    # get the value from args
    value = args[0]
    # get method definition from args
    method = args[1]
    # get sequences
    sequences = args[2:]

    # if it's a str
    if isinstance(value, str):
        # loop the sequence
        for sequence in sequences:
            # generate code
            code = f'"{value}".{method}({sequence})'
            # compile the code
            compiled_code = compile(code, "", "eval")
            # evaluate the compiled code
            value = eval(compiled_code)
        # return the value
        return value
    else:
        # loop the sequence
        for sequence in sequences:
            # generate code
            code = f'value.{method}({sequence})'
            # compile the code
            compiled_code = compile(code, "", "exec")
            # execute the compiled code
            exec(compiled_code)
        # return the value
        return value


# Question 2
def padded_broadcasting(func, a, b, pad=1):
    '''
    Definition:
    Input:
        :param func: function name
        :param a: value of a
        :param b: value of b
        :param pad: padding value
    Output:
        :return: result
    '''
    # get dim of a and b
    dima = a.shape
    dimb = b.shape
    # initialize max dimention array
    max_dim = [0 for i in range(max(len(a.shape), len(b.shape)))]
    # loop, calculate max dimention
    for i in range(len(max_dim)):
        max_dim[i] = max(dima[i], dimb[i])
    # convert to tuple
    max_dim = tuple(max_dim)
    # add values if not enough
    while len(dima) < len(max_dim):
        dima.append(0)
    # add values if not enough
    while len(dimb) < len(max_dim):
        dimb.append(0)
    # make pad
    pad_a = [max_dim[i] - dima[i] for i in range(len(max_dim))]
    pad_b = [max_dim[i] - dimb[i] for i in range(len(max_dim))]
    pad_with = [[0, x] for x in pad_a]
    a = np.pad(a, pad_width=pad_with, constant_values=pad)
    pad_with = [[0, x] for x in pad_b]
    b = np.pad(b, pad_width=pad_with, constant_values=pad)
    # do function
    c = func(a, b)
    return c


# Question 3
def txtanalyser(fname, t, f, sel):
    '''
    analysis the txt
    :param fname: filename
    :param t: word
    :param f: function
    :param sel: count or find
    :return: result
    '''
    # read into data
    data = np.loadtxt(fname, delimiter='\t', skiprows=1, dtype=str)

    # split data into minutes and commentary
    minutes = data[:, 0]
    commentary = data[:, 1]

    # calculate word occurrences
    word_occurrences = np.char.count(commentary, t)

    if sel == 'count':
        res = 0
        # loop commentary
        for x in commentary:
            # if the whole word contains
            if t in str(x).split():
                # counter increase 1
                res += 1
        return float(res)
    elif sel == 'find':
        # Find the indices where the target word occurs
        indices = np.where(word_occurrences > 0)

        # Extract the minutes where the target word occurs
        minutes_with_word = minutes[indices]

        # Convert minutes to integers and calculate the specified function
        result = f(minutes_with_word.astype(int))

        return result
    


# Question 4
def clean_text(commentary):
    '''
    remove the punctuation
    turn spanish into english
    :param commentary:
    :return:the cleaned commentary
    '''
    # Remove punctuation and handle special characters
    cleaned_commentary = re.sub(r'[^\w\s]', '', commentary)
    cleaned_commentary = cleaned_commentary.replace('á', 'a').replace('é', 'e').replace('í', 'i').replace('ó','o').replace('ú', 'u').replace('ñ', 'n').replace('Ñ', 'N')
    return cleaned_commentary

def is_sorted_str(txt1, txt2, check_ties):
    # if check ties is True
    if check_ties:
        # loop each letter in two word
        for i in range(min(len(txt1), len(txt2)) - 1):
            # if it's same, pass it
            if txt1[i] == txt2[i]:
                pass
            # if txt1 is greater, make it False
            elif txt1[i] > txt2[i]:
                return False
            # else make it True
            else:
                return True
        return len(txt1) <= len(txt2)
    else:
        for i in range(min(len(txt1), len(txt2)) - 1):
            # if it's same, make it False
            if txt1[i] == txt2[i]:
                return False
            # if txt1 is greater, make it False
            elif txt1[i] > txt2[i]:
                return False
            # else make it True
            else:
                return True
        return len(txt1) < len(txt2)

def is_sorted_sp(sp, check_ties):
    # loop each word
    for i in range(len(sp) - 1):
        # see if it's sorted
        if not is_sorted_str(sp[i], sp[i + 1], check_ties):
            return False
    return True

def find_alphabetical_order(fname, check_ties=True):
    # read into data
    data = np.loadtxt(fname, delimiter='\t', skiprows=1, dtype=str)

    # split data into minutes and commentary
    minutes = data[:, 0]
    commentary = data[:, 1]

    # clean the data
    a = [[txt, clean_text(txt)] for txt in commentary]
    b = []
    # loop
    for origin, txt in a:
        sp = txt.split()
        sp = [x.lower() for x in sp]
        # take the first letter
        b.append([origin, sp])
    c = []
    # judge if the letter is sorted
    for origin, sp in b:
        if is_sorted_sp(sp, check_ties):
            c.append([origin, txt, sp])
    # get the filtered list
    return [x for x, y, z in c]


# Question 4 - Commentary languages (1 Mark): How many English comments did we find:
# My answer is : There are 5 English comments 


# Question 4 - Name 3 footballers (1 Mark): Name 3 footballers that appear in comments:
# My answer is : 1. Dimitris Giannoulis 2. Armando Broja 3. Illan Meslier 


# Question 5
def pd_query():
    # Load the superhero data from the CSV file
    df = pd.read_csv('superheros.csv')

    # superheros who are good, has greater than 80 intelligence but lower than 20 speed.
    q1 = df[(df['Alignment'] == 'good') & (df['Intelligence'] > 80) & (df['Speed'] < 20)].reset_index()

    #  average values of each column for superheros with durability value of 120.
    q2 = df[df['Durability'] == 120].iloc[:, 2:].mean()

    # average Total values of good and bad superheros.
    q3 = df.groupby('Alignment')['Total'].mean()[['good', 'bad']]

    # names of superheros who has at least 3 column values better than Iron Man.
    q4 = df[df.iloc[:, 2:].gt(df.loc[df['Name'] == 'Iron Man'].iloc[0, 2:], axis=1).sum(axis=1) >= 3].reset_index()

    return q1, q2, q3, q4