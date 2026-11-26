using Microsoft.Data.SqlClient;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms.TimeSeries;

namespace MLNET;

internal sealed class Program
{
    private const string Connection = @"Server=RPM-DEV\SQLEXPRESS;Database=BB84.Web.TimeTracking;Integrated Security=true;TrustServerCertificate=true";
    private const string SqlStatement = """
    SELECT CAST(BreakMinutes AS REAL) AS BreakMinutes,
           CAST(DATEDIFF(minute, '00:00', [Start]) AS REAL) AS StartMinutes,
           CAST(DATEDIFF(minute, [Start], [End]) AS REAL) AS DurationMinutes,
           CAST(DATEDIFF(day, '19000101', [Date]) % 7 AS REAL) AS Weekday -- 0 = Monday, independent of DATEFIRST
    FROM [TimeTracking].[TimeRecordings]
    WHERE BreakMinutes IS NOT NULL AND [Start] IS NOT NULL AND [End] IS NOT NULL
    ORDER BY [Date]
    """;

    private const int Holdout = 60;

    private static void Main(string[] args)
    {
        var ml = new MLContext(seed: 0);
        var data = ml.Data.CreateDatabaseLoader<DayInput>()
            .Load(new DatabaseSource(SqlClientFactory.Instance, Connection, SqlStatement));
        var all = ml.Data.CreateEnumerable<DayInput>(data, reuseRowObject: false).ToArray();

        var breakMinutes = Forecast(ml, data, all, nameof(DayInput.BreakMinutes), x => x.BreakMinutes);
        var start = Forecast(ml, data, all, nameof(DayInput.StartMinutes), x => x.StartMinutes);
        var duration = Forecast(ml, data, all, nameof(DayInput.DurationMinutes), x => x.DurationMinutes);

        Console.WriteLine($"Next BreakMinutes: {breakMinutes.Forecast[0]:F0} (95%: {breakMinutes.Lower[0]:F0}–{breakMinutes.Upper[0]:F0})");
        Console.WriteLine($"Next Start: {ToTime(start.Forecast[0])} (95%: {ToTime(start.Lower[0])}–{ToTime(start.Upper[0])})");
        Console.WriteLine($"Next Duration: {duration.Forecast[0]:F0} min (95%: {duration.Lower[0]:F0}–{duration.Upper[0]:F0})");
        // Intervals of Start and Duration do not add up to an End interval, so End gets the point value only.
        Console.WriteLine($"Next End: {ToTime(start.Forecast[0] + duration.Forecast[0])}");
    }

    // Backtests one column, then trains on all rows and forecasts the next value.
    private static DayForecast Forecast(MLContext ml, IDataView data, DayInput[] all, string column, Func<DayInput, float> value)
    {
        Backtest(ml, all, column, value);
        BacktestRegression(ml, all, column, value);
        return CreatePipeline(ml, column, all.Length)
            .Fit(data)
            .CreateTimeSeriesEngine<DayInput, DayForecast>(ml)
            .Predict();
    }

    // Rolling one-step-ahead over the last Holdout rows: each step forecasts, then feeds the actual value in.
    private static void Backtest(MLContext ml, DayInput[] all, string column, Func<DayInput, float> value)
    {
        var train = all[..^Holdout];
        var test = all[^Holdout..];

        var engine = CreatePipeline(ml, column, train.Length)
            .Fit(ml.Data.LoadFromEnumerable(train))
            .CreateTimeSeriesEngine<DayInput, DayForecast>(ml);

        double sum = train.Sum(value), ssaError = 0, meanError = 0, lastError = 0;
        var count = train.Length;
        var last = value(train[^1]);
        var forecast = engine.Predict().Forecast[0];

        foreach (var actual in test)
        {
            var observed = value(actual);
            ssaError += Math.Abs(forecast - observed);
            meanError += Math.Abs(sum / count - observed);
            lastError += Math.Abs(last - observed);
            sum += observed;
            count++;
            last = observed;
            forecast = engine.Predict(actual).Forecast[0];
        }

        Console.WriteLine($"Backtest {column}, last {Holdout} rows, MAE: SSA {ssaError / Holdout:F2} min, mean baseline {meanError / Holdout:F2} min, last-value baseline {lastError / Holdout:F2} min");
    }

    // Linear regression on weekday (one-hot), previous value and mean of the previous 5 values, same holdout rows as Backtest.
    private static void BacktestRegression(MLContext ml, DayInput[] all, string column, Func<DayInput, float> value)
    {
        const int Lags = 5;
        var rows = Enumerable.Range(Lags, all.Length - Lags).Select(i => new RegressionRow
        {
            Weekday = all[i].Weekday,
            Last = value(all[i - 1]),
            Mean5 = all[(i - Lags)..i].Average(value),
            Label = value(all[i])
        }).ToArray();

        var pipeline = ml.Transforms.Categorical.OneHotEncoding(nameof(RegressionRow.Weekday))
            .Append(ml.Transforms.Concatenate("Features", nameof(RegressionRow.Weekday), nameof(RegressionRow.Last), nameof(RegressionRow.Mean5)))
            .Append(ml.Transforms.NormalizeMinMax("Features"))
            .Append(ml.Regression.Trainers.Sdca());

        // shortcut: trained once on the train rows, not refit per step like SSA's state update; refit if the gap matters.
        var model = pipeline.Fit(ml.Data.LoadFromEnumerable(rows[..^Holdout]));
        var metrics = ml.Regression.Evaluate(model.Transform(ml.Data.LoadFromEnumerable(rows[^Holdout..])));

        Console.WriteLine($"Regression {column}, last {Holdout} rows, MAE: {metrics.MeanAbsoluteError:F2} min");
    }

    private static SsaForecastingEstimator CreatePipeline(MLContext ml, string column, int trainSize) =>
        ml.Forecasting.ForecastBySsa(
            outputColumnName: nameof(DayForecast.Forecast),
            inputColumnName: column,
            windowSize: 5,          // Mon–Fri cycle
            seriesLength: 20,
            trainSize: trainSize,
            horizon: 1,             // next work day
            confidenceLevel: 0.95f,
            confidenceLowerBoundColumn: nameof(DayForecast.Lower),
            confidenceUpperBoundColumn: nameof(DayForecast.Upper));

    private static string ToTime(float minutes) => TimeSpan.FromMinutes(Math.Round(minutes)).ToString(@"hh\:mm");

    internal sealed class DayInput
    {
        public float BreakMinutes { get; set; }
        public float StartMinutes { get; set; }
        public float DurationMinutes { get; set; }
        public float Weekday { get; set; }
    }

    internal sealed class RegressionRow
    {
        public float Weekday { get; set; }
        public float Last { get; set; }
        public float Mean5 { get; set; }
        public float Label { get; set; }
    }

    internal sealed class DayForecast
    {
        public float[] Forecast { get; set; } = [];
        public float[] Lower { get; set; } = [];
        public float[] Upper { get; set; } = [];
    }
};
