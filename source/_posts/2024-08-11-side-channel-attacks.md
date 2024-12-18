---
layout: post
title: Understanding Differential Power Analysis (DPA)
date: 2024-08-11 04:04:06 -0400
tags: 
- side-channel attacks
- crypto
- blogpost
---

This is an old writeup that I'm cleaning up and uploading in the next day or two. Please be patient! 


I wanted to get a hands-on introduction to power-based side-channel analysis. An implementation of AES-ECB was created and I took a set of power traces during the execution, with the associated input/output values of AES, and I performed an attack to extract the 128-bit secret key via DPA. 

The power traces contained 10,000 AES-ECB executions, with random input and a fixed, secret key.

# Power Trace Analysis

Let's take a look at the power trace itself, and the data we have in it. 

The peaks in the trace (both peak and trough) are the maximum leak points. 
These peaks indicate the specific time for an adversary to retrieve the secret key. 
These signify the 11 addRoundKey() functions in AES-128 bit encryption.

![Leak Point Graph](/assets/dpa/1/leakpoints.png)

# Attack

## Analysing the first power trace

Let’s execute the DPA attack using Pearson’s correlation test. We'll take a look at the power trace, corresponding to the first encryption operation.
1. Think of an easy target operation for the DPA 
2. Consider the number of bits to be estimated by DPA at a time.

![Leak Point Graph](/assets/dpa/1/correlation.png)

The AES implementation was done on an FPGA, therefore, we will use a hamming distance power model. For each input byte, the FPGA will store the initial input, then we will perform the addRoundKey operation.

## Breaking Apart the Maximum Correlation into 2 

![Maximum Correlation](/assets/dpa/1/positive-negative.png)

Both lines are symmetrical to each other. The negative graph is more likely to be the correct
guess, because it looks more like the power trace. 
This is key = 0x00.

## Finding the Maximum Leak Point

In time domain, the maximum leak point is t = 43. This is a plot of the "evolution" of key hypotheses for all measurements at t=43

![evolution](/assets/dpa/1/evolution.png)