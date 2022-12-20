from sklearn.ensemble import GradientBoostingClassifier
gbm_model = GradientBoostingClassifier()
gbm_fit = gbm_model.fit(X_train, Y_train)
gbm_pred = gbm_fit.predict(X_test)
