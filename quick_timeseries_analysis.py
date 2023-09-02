import datetime
import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from causalimpact import CausalImpact as CI
from plotly import express, graph_objects
from plotly.subplots import make_subplots
from prophet import Prophet as FP
from prophet.utilities import regressor_coefficients
from sklearn.preprocessing import MinMaxScaler


class ModelWrapper:
    """
    Custom Wrapper for popular models for TimeSeries analysis, including but not limited to:
        - Google Causal Impact https://pypi.org/project/causalimpact/
        - Facebook Prophet https://facebook.github.io/prophet/

    Parameters
    ----------
    df: DataFrame containg target variable with `y` name and time frame variable with `date` name
    start_date: the start date of experiment with the format YYYY-MM-DD
    test_days: the number of last days in which the experiment was run
    end_date: the end date of experiment with the format YYYY-MM-DD
    date: timeframe column name
    y: target column name

    Attributes
    ----------
    explore: the chart exploring pre-experiment correlations
    show: visualization of the trained model, please notet that you need to run the model first
    """

    def __init__(
        self,
        df: pd.DataFrame,
        start_date: Optional[str] = None,
        test_days: Optional[int] = None,
        end_date: Optional[str] = None,
        date: str = "date",
        y: str = "y",
    ) -> None:

        self.df = df.copy()
        self.date = date
        self.y = y

        if start_date:
            self.start_dt = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
        elif test_days:
            if pd.core.dtypes.common.is_datetime64_dtype(self.df[self.date].dtype):
                self.start_dt = self.df[self.date].max().date() - datetime.timedelta(
                    days=test_days
                )
            else:
                self.start_dt = datetime.datetime.strptime(
                    self.df[self.date].max(), "%Y-%m-%d"
                ).date() - datetime.timedelta(days=test_days)
        else:
            raise ValueError("You must specify start_date or test_days variable")

        self.end_dt = (
            datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
            if end_date
            else None
        )
        self.x = list(self.df.columns.difference([self.date, self.y]).values)
        self._preprocess()

    def _preprocess(self):
        self.df[self.date] = pd.to_datetime(self.df[self.date]).dt.date
        self.df = (
            self.df[[self.y, self.date] + self.x].set_index(self.date).sort_index()
        )
        for column in self.df.columns[self.df.dtypes == "object"]:
            try:
                self.df[column] = self.df[column].astype(float)
            except ValueError:
                raise ValueError("All DataFrame columns except Date must be numeric")

    @staticmethod
    def _save(figure, title: str) -> None:
        figure.write_html(re.sub('[-:<>|\/\*\?"\\\\ ]+', "_", title.lower()) + ".html")

    def explore(
        self,
        scale: bool = True,
        title: str = "pre-experiment correlation",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> None:
        """
        Plot the dynamic of pre-experiment correlation

        Parameters
        ----------
        scale: whether the samples should be scaled to [0, 1] interval
        title: the title of the chart
        x_label: label for X axis
        y_label: label for Y axis
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """

        data = self.df.copy()

        if scale:
            scaler = MinMaxScaler(feature_range=(0, 1))
            data[data.columns] = scaler.fit_transform(data)

        corr = data[data.index < self.start_dt].corr().iloc[0, 1:]

        data = pd.melt(
            data,
            value_vars=list(data.columns),
            var_name="variable",
            value_name="value",
            ignore_index=False,
        )

        chart_title = (
            title
            + f"""<br><sup>{', '.join(
            [f"{feature}: {round(value, 2)}"
            for feature, value in zip(self.x, corr)]
        )}</sup></br>"""
        )

        figure = express.line(
            data,
            x=data.index,
            y="value",
            color="variable",
            height=height,
            width=width,
            title=chart_title,
        )

        figure.add_vline(
            x=self.start_dt, line_width=2, line_dash="dash", line_color="white"
        )
        if self.end_dt:
            figure.add_vline(
                x=self.end_dt, line_width=2, line_dash="dash", line_color="white"
            )

        figure.update_traces(
            hovertemplate="%{y}"
            if pd.core.dtypes.common.is_integer_dtype(data["value"])
            else "%{y:.3f}"
        )
        figure.update_xaxes(title_text=x_label if x_label else "Date")
        figure.update_yaxes(
            title_text=y_label
            if y_label
            else "Scaled Axis"
            if scale
            else "Original Axis"
        )

        figure.update_layout(
            title={
                "x": 0.5,
            },
            legend={"x": 0.05, "y": 1.05, "orientation": "h", "title": None},
            hovermode="x",
            template="plotly_dark",
            xaxis=dict(hoverformat="%a, %b %d, %Y"),
        )

        if save:
            self._save(figure, title)

        figure.show()

    @staticmethod
    def _add_chart(
        figure,
        data: pd.DataFrame,
        titles: list,
        y: str,
        name: str,
        row: int,
        actual: bool = False,
        y_label: Optional[str] = None,
    ) -> None:
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y],
                name=name,
                hovertemplate="%{y:.3f}",
                line={"color": "white"},
                legendgroup=f"{row}",
                legendgrouptitle={"text": titles[row]},
                connectgaps=True,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y + "_upper"],
                name="Upper bound",
                hovertemplate="%{y:.3f}",
                line={"color": "deepskyblue", "width": 0.5},
                legendgroup=f"{row}",
                connectgaps=True,
            ),
            row=row,
            col=1,
        )
        figure.add_trace(
            graph_objects.Scatter(
                x=data.index,
                y=data[y + "_lower"],
                name="Lower bound",
                hovertemplate="%{y:.3f}",
                fill="tonexty",
                line={"color": "deepskyblue", "width": 0.5},
                legendgroup=f"{row}",
                connectgaps=True,
            ),
            row=row,
            col=1,
        )

        figure.update_yaxes(
            title_text="" if not y_label else "% Effect" if row == 4 else y_label
        )

        if actual:
            figure.add_trace(
                graph_objects.Scatter(
                    x=data.index,
                    y=data["response"],
                    name="Actual",
                    hovertemplate="%{y}"
                    if pd.core.dtypes.common.is_integer_dtype(data["response"])
                    else "%{y:.3f}",
                    line={"color": "red"},
                    legendgroup=f"{row}",
                    connectgaps=True,
                ),
                row=row,
                col=1,
            )
        else:
            figure.add_hline(y=0, line_width=1, line_color="white", row=row, col=1)

    def show(
        self,
        keep_n_prior_days: Optional[int] = None,
        title: str = "Causal Impact",
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> None:
        """
        Plot the trained model results

        Parameters
        ----------
        keep_n_prior_days: specify the exact number of pre-experiment days you want to keep, skip if you want to show all the time frame
        title: the title of the chart
        x_label: label for X axis
        y_label: label for Y axis
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """
        try:
            if keep_n_prior_days:
                data = self.result[
                    self.result.index
                    > self.start_dt - datetime.timedelta(days=keep_n_prior_days)
                ]
            else:
                data = self.result.iloc[1:]
        except AttributeError:
            raise AttributeError(
                "To show the results run the model first, use .run() method"
            )

        titles = [
            "Model Overview",
            "Actual vs Expected",
            "Effect Size: Actual - Expected",
            "Cumulative Effect",
            "Effect Relative to Expected",
        ]

        if isinstance(self, CausalImpact):
            figure = make_subplots(
                rows=4, cols=1, shared_xaxes=True, subplot_titles=titles[1:]
            )
            for y, name, row in zip(
                ["point_pred", "point_effect", "cum_effect", "rel_effect"],
                [
                    "Expected Values",
                    "Effect Size",
                    "Cumulative Effect",
                    "Relative Effect",
                ],
                range(1, 5),
            ):
                self._add_chart(
                    figure,
                    data,
                    titles,
                    y=y,
                    name=name,
                    row=row,
                    actual=(row == 1),
                    y_label=y_label,
                )
        elif isinstance(self, Prophet):
            row = 1
            figure = make_subplots()
            self._add_chart(
                figure,
                data,
                titles,
                y="yhat",
                name="Expected Value",
                row=row,
                actual=True,
                y_label=y_label,
            )

        figure.update_xaxes(title_text=x_label if x_label else "Date", row=row, col=1)

        figure.add_vline(
            x=self.start_dt, line_width=2, line_dash="dash", line_color="white"
        )
        if self.end_dt:
            figure.add_vline(
                x=self.end_dt, line_width=2, line_dash="dash", line_color="white"
            )

        figure.update_layout(
            title={
                "x": 0.5,
                "text": title,
            },
            width=width,
            height=height,
            hovermode="x",
            template="plotly_dark",
            legend={
                "x": 0.0,
                "y": -0.2,
                "orientation": "h",
                "groupclick": "toggleitem",
                "traceorder": "grouped",
            },
            xaxis=dict(hoverformat="%a, %b %d, %Y"),
        )

        if save:
            self._save(figure, title)

        figure.show()


class CausalImpact(ModelWrapper):
    def run(
        self, nseasons: int = 7, season_duration: int = 1, alpha: float = 0.05, **kwargs
    ) -> pd.DataFrame:
        """
        Run causal impact analysis

        Parameters
        ----------
        nseasons: Period of the seasonal components.
            In order to include a seasonal component, set this to a whole number greater than 1.
            For example, if the data represent daily observations, use 7 for a day-of-week component.
            This interface currently only supports up to one seasonal component.
        season_duration: Duration of each season, i.e., number of data points each season spans.
            For example, to add a day-of-week component to data with daily granularity, use model_args = list(nseasons = 7, season_duration = 1).
            To add a day-of-week component to data with hourly granularity, set model_args = list(nseasons = 7, season_duration = 24).
        alpha : Desired tail-area probability for posterior intervals. Defaults to 0.05, which will produce central 95% intervals

        Other Parameters
        ----------------
        **kwargs : model_args variables, available options:
            ndraws: number of MCMC samples to draw.
                More samples lead to more accurate inferences. Defaults to 1000.
            nburn: number of burn in samples.
                This specifies how many of the initial samples will be discarded. defaults to 10% of ndraws.
            standardize_data: whether to standardize all columns of the data before fitting the model.
                This is equivalent to an empirical Bayes approach to setting the priors.
                It ensures that results are invariant to linear transformations of the data.
            prior_level_sd: prior standard deviation of the Gaussian random walk of the local level.
                Expressed in terms of data standard deviations. Defaults to 0.01.
                A typical choice for well-behaved and stable datasets with low residual volatility after regressing out known predictors.
                When in doubt, a safer option is to use 0.1, as validated on synthetic data,
                although this may sometimes give rise to unrealistically wide prediction intervals.
            dynamic_regression: whether to include time-varying regression coefficients.
                In combination with a time-varying local trend or even a time-varying local level,
                this often leads to overspecification, in which case a static regression is safer. Defaults to FALSE.
        """

        data = self.df.copy().reset_index()
        prior = [
            data.index.min(),
            int(data[data[self.date] < self.start_dt].index.max()),
        ]
        posterior = [
            int(
                data[
                    data[self.date] >= (self.end_dt if self.end_dt else self.start_dt)
                ].index.min()
            ),
            data.index.max(),
        ]
        data.drop(columns=[self.date], inplace=True)
        self.ci = CI(
            data,
            prior,
            posterior,
            model_args={
                "nseasons": nseasons,
                "season_duration": season_duration,
                **kwargs,
            },
            alpha=alpha,
        )
        self.ci.run()

    def summary(self, format: str = "summary") -> None:
        """
        Print the summary for Causal Impact Analysis model

        Parameters
        ----------
        format: can be 'summary' to return a table or 'report' to return a natural language description
        """
        try:
            self.ci.summary(format)
        except AttributeError:
            raise AttributeError(
                "To get the summary run the model first, use .run() method"
            )
        self.result = self.ci.inferences.set_index(self.df.index)
        for suffix in ["", "_lower", "_upper"]:
            self.result["rel_effect" + suffix] = (
                self.result["point_effect" + suffix] / self.result["point_pred"]
            )


class Prophet(ModelWrapper):
    def run(
        self,
        growth: str = "linear",
        weekly_seasonality: bool = True,
        monthly_seasonality: bool = True,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "additive",
        country_holidays: Optional[str] = None,
        outliers: Optional[List[Tuple[str]]] = None,
        floor: Optional[int] = None,
        cap: Optional[int] = None,
        alpha: float = 0.05,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Run time-series forecasting

        Parameters
        ----------
        growth: String 'linear', 'logistic' or 'flat' to specify a linear, logistic or flat trend.
        weekly_seasonality: Fit weekly seasonality.
        monthly_seasonality: Fit monthly seasonality.
        yearly_seasonality: Fit yearly seasonality.
        seasonality_mode: 'additive' (default) or 'multiplicative'.
        country_holidays: country code (e.g. 'RU') of the country whose holiday are to be considered
        outliers: list of time intervals (start date, end date) with the format YYYY-MM-DD where there are outliers
        floor: minimum allowed value for the target variable. It's particulary useful with "logistic" growth type
        cap: maximum allowed value for the target variable. It's particulary useful with "logistic" growth type
        alpha: 1 - width of the uncertainty intervals provided for the forecast.

        Other Parameters
        ----------------
        **kwargs : model_args variables, to reveal the whole list follow the Prophet documentation, for example:
            n_changepoints: Number of potential changepoints to include. Not used
                if input `changepoints` is supplied. If `changepoints` is not supplied,
                then n_changepoints potential changepoints are selected uniformly from
                the first `changepoint_range` proportion of the history.
            changepoint_range: Proportion of history in which trend changepoints will
                be estimated. Defaults to 0.8 for the first 80%. Not used if
                `changepoints` is specified.
            changepoint_prior_scale: Parameter modulating the flexibility of the
                automatic changepoint selection. Large values will allow many
                changepoints, small values will allow few changepoints.
        """

        data = (
            self.df.copy()
            .reset_index()
            .rename(columns={self.date: "ds", self.y: "y"})
            .sort_values("ds")
        )

        if cap:
            data["cap"] = cap
        if floor:
            data["floor"] = floor

        train, test = (
            data[data["ds"] < self.start_dt],
            data[data["ds"] >= self.start_dt],
        )

        self.model = FP(
            growth=growth,
            weekly_seasonality=weekly_seasonality,
            yearly_seasonality=yearly_seasonality,
            seasonality_mode=seasonality_mode,
            interval_width=1 - alpha,
            **kwargs,
        )

        if monthly_seasonality:
            self.model.add_seasonality(name="monthly", period=30.5, fourier_order=5)

        if country_holidays:
            self.model.add_country_holidays(country_name=country_holidays)

        if outliers:
            for pair in outliers:
                train.loc[
                    (
                        train["ds"]
                        > datetime.datetime.strptime(pair[0], "%Y-%m-%d").date()
                    )
                    & (
                        train["ds"]
                        < datetime.datetime.strptime(pair[1], "%Y-%m-%d").date()
                    ),
                    "y",
                ] = None

        for feature in self.x:
            self.model.add_regressor(feature)

        self.model.fit(train)

        future = self.model.make_future_dataframe(periods=test.shape[0])

        self.result = self.model.predict(
            future.set_index("ds").join(data.set_index("ds")).reset_index()
        )
        self.result["ds"] = self.result["ds"].dt.date
        self.result = (
            self.result.set_index("ds")
            .join(data[["ds", "y"]].set_index("ds"))
            .rename(columns={"y": "response"})
        )

    def summary(
        self,
        width: int = 900,
        height: int = 600,
        save: bool = False,
    ) -> pd.DataFrame:
        """
        Plot the regressors statistics: Coefficients, Impact and Impact Share

        The estimated beta coefficient for each regressor roughly represents the increase
        in prediction value for a unit increase in the regressor value.
        Note that the coefficients returned are always on the scale of the original data
        In addition the credible interval for each coefficient is also returned,
        which can help identify whether each regressor is “statistically significant”.

        On the basis of `seasonality_mode` the model looks like:
            Additive: y(t) ~ trend(t) + seasonality(t) + beta * regressor(t)
            Multiplicative: y(t) ~ trend(t) * ( 1 + seasonality(t) + beta * regressor(t) )

        Therefore, the incremental impact are:
            Additive: increasing the value of the regressor by a unit leads to an increase in y(t) by beta units
            Multiplicative: increasing the value of the regressor by a unit leads to increase in y(t) by beta * trend(t) units

        The Impact is the product of incremental impact(t) * regressor(t) and finally, Share is the percentage of absolute Impact

        Parameters
        ----------
        width: the width of the chart
        height: the height of the chart
        save: whether you want to save the chart as HTML
        """

        try:
            data = regressor_coefficients(self.model)
        except AttributeError:
            raise AttributeError(
                "To get the summary run the model first, use .run() method"
            )

        last_day_data, last_day_result = self.df.iloc[-1, :], self.result.iloc[-1, :]

        data["incremental_impact"] = data["coef"] * data["regressor_mode"].apply(
            lambda x: last_day_result["trend"] if x == "multiplicative" else 1
        )
        data["impact"] = data.apply(
            lambda x: x["incremental_impact"] * last_day_data[x["regressor"]], axis=1
        )
        data["share"] = round(
            100 * np.abs(data["impact"]) / np.sum(np.abs(data["impact"])), 2
        )

        def plot_bar(data, y, title):
            figure = express.bar(
                data,
                x="regressor",
                y=y,
                color="regressor",
                color_discrete_sequence=express.colors.sequential.Jet,
            )
            figure.add_hline(y=0, line_width=1, line_color="white")
            figure.update_xaxes(title_text=None)
            figure.update_layout(
                title={
                    "x": 0.5,
                    "text": title,
                },
                width=width,
                height=height,
                hovermode="x",
                template="plotly_dark",
                showlegend=False,
            )
            figure.update_traces(hovertemplate="%{y:.3f}")

            if save:
                self._save(figure, title)

            figure.show()

        for y, title in zip(["coef", "impact"], ["Coefficients", "Impact"]):
            plot_bar(data.sort_values(by="coef", ascending=False), y, title)

        pie = express.pie(
            data.sort_values(by="coef", ascending=False),
            color_discrete_sequence=express.colors.sequential.Jet,
            values="share",
            names="regressor",
            color="regressor",
        )

        pie.update_layout(
            title={
                "x": 0.5,
                "text": "Impact Share",
            },
            width=width,
            height=height,
            hovermode="x",
            template="plotly_dark",
            showlegend=False,
        )

        pie.update_traces(
            hovertemplate="Regressor: %{label}<br>Share: %{value:.2f}%</br>",
            marker=dict(line=dict(color="black", width=3)),
        )

        if save:
            self._save(pie, "Impact Share")

        pie.show()
