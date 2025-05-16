import os
import json

import math

import numpy as np
import scipy.stats as stats
import matplotlib
#import scipy.stats.distrubutions

matplotlib.use('Agg')  # Use a non-interactive backend
import matplotlib.pyplot as plt
from settings import max_visual_distance
import seaborn as sns
from settings import debug

def extract_tags_from_json(directory_path, tag):
    extracted_data_for_trial = []

    for filename in os.listdir(directory_path):
        if filename.endswith(".json"):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r') as json_file:
                try:
                    data = json.load(json_file)
                    # add code inhere
                    #
                    #

                except json.JSONDecodeError:
                    print(f"Error decoding JSON from file: {filename}")

    return extracted_data_for_trial


def calculate_distance(point1, point2):
    # Unpack the coordinates
    x1, y1, z1 = point1['x'], point1['y'], point1['z']
    x2, y2, z2 = point2['x'], point2['y'], point2['z']

    # Calculate the distance using the Euclidean distance formula
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    return distance

def extract_repetitive_and_nested_tags(json_file_path, outer_tag, nested_tag_path, nested_tag_path_2):
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    values = []
    values_2 = []
    nested_values_gaze = []
    nested_values_intersection = []
    distance = []

    if isinstance(data, dict):
        for item in data.get('Frames', []):
            if outer_tag in item:
                values.append(item[outer_tag])
            nested_tag = item

            for tag in nested_tag_path:
                if tag in nested_tag:
                    nested_tag = nested_tag[tag]
                else:
                    nested_tag = None
                    break
            if nested_tag is not None:
                nested_values_gaze.append(nested_tag)

    if isinstance(data, dict):
        for item in data.get('Frames', []):
            if outer_tag in item:
                values_2.append(item[outer_tag])
            nested_tag = item

            for tag in nested_tag_path_2:
                if tag in nested_tag:
                    nested_tag = nested_tag[tag]
                else:
                    nested_tag = None
                    break
            if nested_tag is not None:
                nested_values_intersection.append(nested_tag)


    if len(nested_values_gaze) == len(nested_values_intersection):
        #print("List are of equal size")
        for point1, point2 in zip(nested_values_gaze, nested_values_intersection):
            distance.append(calculate_distance(point1, point2))
    else:
        #print("List are having diff sizes")
        print(len(nested_values_gaze), len(nested_values_intersection))

    return values, nested_values_gaze, nested_values_intersection, distance

def cal_hist(values):
    values = [value for value in values if value <= max_visual_distance]

    # Plot the histogram
    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.hist(values, bins=10, color='skyblue', edgecolor='black')
    plt.title('Histogram')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Plot the KDE
    plt.subplot(1, 3, 2)
    sns.kdeplot(values, bw_adjust=0.5, color='red', fill=True)
    plt.title('Kernel Density Estimate')
    plt.xlabel('Value')
    plt.ylabel('Density')

    # Generate KDE values using Seaborn's kdeplot and retrieve the y values
    kde = sns.kdeplot(values, bw_adjust=0.5)
    x, y = kde.get_lines()[0].get_data()

    # Take the logarithm of the KDE values
    log_y = np.log(y)

    plt.subplot(1, 3, 3)
    plt.plot(x, log_y, color='red')
    plt.fill_between(x, log_y, alpha=0.2, color='red')
    plt.title('Logarithm of KDE')
    plt.xlabel('Values')
    plt.ylabel('Log Density')
    plt.legend()
    # Save the plot to a file without displaying it
    plt.savefig(os.path.join(directory_path, 'kde_distribution_plot.png'))


def cal_pdf_v2(values, plot_url=''):
    values = [value for value in values if value <= max_visual_distance]

    # Calculate the mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # Create a normal distribution using the calculated mean and standard deviation
    normal_dist = stats.lognorm(mean, std_dev)


    # Calculate the PDF for each value in the list
    pdf_values = normal_dist.pdf(values)

    inverse_log_pdf_values = np.exp(pdf_values)

    for value, pdf in zip(values, pdf_values):
        print(f"Value: {value}, PDF: {pdf}")

    x = np.linspace(min(values) - 3 * std_dev, max(values) + 3 * std_dev, 1000)
    y = normal_dist.pdf(x)

    plt.plot(x, y, label='Log Normal Distribution')
    plt.scatter(values, pdf_values, color='red', label='PDF Values')
    plt.title('Log Normal Distribution and PDF Values')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim(0, 50)
    # Save the plot to a file without displaying it
    plt.savefig(os.path.join(plot_url, 'normal_distribution_plot.png'))
    # Clear the figure after saving to avoid any conflicts if more plots are created later
    plt.clf()

    plt.plot(x, y, label='Normal Distribution')
    plt.scatter(values, inverse_log_pdf_values, color='red', label='PDF Values')
    plt.title('Normal Distribution and PDF Values')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim(0, 50)
    # Save the plot to a file without displaying it
    plt.savefig(os.path.join(plot_url, 'normal_log_distribution_plot.png'))
    # Clear the figure after saving to avoid any conflicts if more plots are created later
    plt.clf()

    return normal_dist, pdf_values


def cal_pdf(values, plot_url=''):
    values = [value for value in values if value <= max_visual_distance]

    # Calculate the mean and standard deviation
    mean = np.mean(values)
    std_dev = np.std(values)

    # Create a normal distribution using the calculated mean and standard deviation
    normal_dist = stats.norm(mean, std_dev)

    # Calculate the PDF for each value in the list
    pdf_values = normal_dist.pdf(values)

    inverse_log_pdf_values = np.exp(pdf_values)

    x = np.linspace(min(values) - 3 * std_dev, max(values) + 3 * std_dev, 1000)
    y = normal_dist.pdf(x)

    plt.plot(x, y, label='LogNormal')
    plt.scatter(values, pdf_values, color='red', label='PDF_Values')
    plt.title('Log Normal Distribution and PDF Values')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim(0, 50)
    if plot_url == "":
        plot_url = '/home/jovyan/Torte/data_samples/'
    # Save the plot to a file without displaying it
    plt.savefig(os.path.join(plot_url, 'normal_distribution_plot.png'))
    # Clear the figure after saving to avoid any conflicts if more plots are created later
    plt.clf()

    plt.plot(x, y, label='Normal')
    plt.scatter(values, inverse_log_pdf_values, color='red', label='PDF_Values')
    plt.title('Normal Distribution and PDF Values')
    plt.xlabel('Values')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.xlim(0, 50)
    # Save the plot to a file without displaying it
    plt.savefig(os.path.join(plot_url, 'normal_log_distribution_plot.png'))
    # Clear the figure after saving to avoid any conflicts if more plots are created later
    plt.clf()
    if debug:
        print("Norm_dist=", normal_dist)
        print("pfd_val = ", pdf_values)
        print("mean=", mean)
        print("std_dev", std_dev)

    return normal_dist, pdf_values, mean, std_dev