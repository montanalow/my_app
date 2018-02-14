my_app
==========

System Setup
------------

1) Get your system setup

.. code::

  $ lore install

2) Set correct variables in `.env`

.. code::

  $ cp .env.template .env
  $ edit .env

Running
-------

The service runs on {DOMAIN}, to run locally:

.. code::

  $ lore console
  $ lore server

Testing
-------

To run locally:

.. code::

  $ lore test

Training
--------

To train locally:

.. code::

  $ lore notebook

Deploying
---------

.. code::

  $ git push heroku

.. _CircleCI: https://circleci.com/
.. _Domino: https://domino.io/
