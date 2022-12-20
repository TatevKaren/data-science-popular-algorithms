from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=10, random_state=0)
#fitting the model on the training data
rf_fit = rf_model.fit(X_train, Y_train)
#using fitted model to get the predictions for test data
rf_pred = rf_fit.predict(X_test)