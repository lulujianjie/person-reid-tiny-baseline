Triplet Loss
============

Triplet Loss
------------



.. math::
    L = max(D_{an} - D_{ap} + m, 0)


Hard Example Mining
-------------------

.. math::
    L = max(min(D_{an}) - max(D_{ap}) + m, 0)

Harder Example Mining
---------------------

.. math::
    L = max((1 - \alpha)\cdot min(D_{an}) - (1 + \alpha)\cdot max(D_{ap}) + m, 0)