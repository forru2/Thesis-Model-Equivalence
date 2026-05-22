# Defining Functional Behavioural Equivalence
# within Interpretable Models of the
# Rashomon Set

The Rashomon Effect describes a common phenomenon in machine learning,
where multiple models can achieve nearly identical predictive performance on
the same dataset while employing different internal logics. While traditional
model selection relies almost exclusively on accuracy, the existence of a Rashomon
Set (RS) of equally performing models provides a unique opportunity to incor
porate additional evaluation criteria. These criteria, such as fairness, reliability,
and interpretability, allow exploration of the internal behaviour of models. This
project proposes an inter-family, multi-dimensional framework to define and iden
tify Functional Behavioural Equivalence among interpretable models. Moving
beyond simple performance-based evaluation and selection, the research investi
gates whether models within the Rashomon Set also exhibit similar behaviours
across alternative dimensions. We focus on three families of interpretable mod
els: Decision Trees, Linear Regressors, and K-Nearest Neighbors. They are
applied to four tabular classification tasks in sensitive domains like finance and
law enforcement. The methodology involves a structured workflow: first, a di
verse pool of models is trained to ensure broad coverage of the model space and
to enhance the Rashomon Effect; second, a Rashomon Set is constructed based
on generalization performance. Finally, an exploratory equivalence analysis is
conducted using a clustering-based approach. The results provide stakeholders
with a “map” of interchangeable solutions, facilitating the exploration of the
space of well-performing models that go beyond mere prediction accuracy and
are also ethically and behaviourally aligned with human values
