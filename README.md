Code accompanying the paper 

# Increasing the efficiency of Sequential Monte Carlo samplers through the use of near-optimal L-kernels 
# P. L. Green, L. J. Devlin, R. E. Moore, R. J. Jackson, J. Li, S. Maskell
# University of Liverpool

## Code structure 

Code accompanying the first submission of the paper (https://arxiv.org/abs/2004.12838) is in the branch 'v1' and can be accessed using `git checkout v1`. We are keeping the master branch separate so that we can use it to bring in future developments if needed. Future versions will then be created when we develop beyond the original paper. 

## Using the code
If you would like to use the code, it would be really useful if did so by forking the current repository. This just gives us a way of seeing how the code is being used etc. 

## Testing
During development we made a series of tests for verification purposes. To execute the tests install `pytest` and run it in the `tests` folder. 

## Contributions
We actively encourage people to contribute to our code. If you would like to do so, please fork the repository and issue a pull request. We try to write everything in PEP8 notation(ish). If you find any problems, please feel free to raise an `issue` through Github and we'll do our best to get back to you.

## Versions
V1 : originally submitted with first version of the paper (https://arxiv.org/abs/2004.12838).

V2 : now uses abstract methods to make it more clear what the user needs to define.

V3 : added example of the approach applied to a single-degree-of-freedom system.

V4 : now uses abstract classes and random seed is fixed for tests.

V5 : now includes the option for the single step sampling approach, which should help performance for higher dimensional problems. This approach was developed in response to reviewer comments about the first version of the paper. 

V6 : fixes issue where calculation of the normalised weights was altering the log weights

## Contact
Queries to: p.l.green@liverpool.ac.uk
