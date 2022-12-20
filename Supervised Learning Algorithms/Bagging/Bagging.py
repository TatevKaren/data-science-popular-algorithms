from sklearn.ensemble import BaggingClassifier
bagging_model = BaggingClassifier(n_estimators=10, random_state=0)
begging_fit = bagging_model.fit(X_train,Y_train)
begging_pred = begging_fit.predict(X_test)