"""This module contains some utility functions.
"""



# calculate the loss for a single simulation
# treats all runs with same weight regardless of number of observations
def loss_function(experimental_run, concentrations_df):
    losses = []
    for t,observed in zip(experimental_run.observation_times, experimental_run.observations):
        df = concentrations_df.query(f"index == {t}")
        assert len(df) == 1
        simulated = concentrations_df.tail(1)[sugar_abbreviations].iloc[0].to_numpy()
        # print("Simulated:")
        # print(simulated)
        # print("Obeserved:")
        # print(observed)
        # print("diff:")
        # print(simulated-observed)
        # print("rms:")
        loss = rms(simulated-observed)
        # print(loss)
        losses.append(loss)
    # print("LOSSES")
    # print(losses)
    # print(rms(losses))
    return rms(losses)

# root mean square average
def rms(x):
    assert isinstance(x, (list,np.ndarray))
    assert len(x) > 0
    return np.sqrt(np.mean(np.square((x))))