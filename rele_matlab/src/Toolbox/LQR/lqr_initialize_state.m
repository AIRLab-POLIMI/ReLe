function initial_state = lqr_initialize_state(simulator)

initial_state = feval(simulator);

return
