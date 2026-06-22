import { useCallback, useEffect, useMemo, useState } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

type BaseDraw = {
  zx: number;
  u: number;
};

type Person = {
  id: number;
  height: number;
  weight: number;
};

type RegressionLine = {
  intercept: number;
  slope: number;
};

type OlsSummary = RegressionLine & {
  interceptSe: number;
  slopeSe: number;
};

type MomentSummary = RegressionLine & {
  meanX: number;
  meanY: number;
  varX: number;
  covXY: number;
};

type PlotDomain = {
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
};

type RegressionPlotProps = {
  ariaLabel: string;
  clipId: string;
  domain: PlotDomain;
  meanPoint: { x: number; y: number };
  populationDots: Person[];
  populationLine: RegressionLine;
  recentLines?: RegressionLine[];
  sampleLine?: RegressionLine | null;
  samplePoints?: Person[];
  showSampleLegend?: boolean;
};

type MathInlineProps = {
  tex: string;
};

type Speed = 'slow' | 'medium' | 'fast';

const NPOP = 10_000;
const MAX_POP_DOTS = 2_500;
const MAX_RECENT_LINES = 40;
const SPEED_MS: Record<Speed, number> = {
  slow: 900,
  medium: 350,
  fast: 120,
};
const SVG_WIDTH = 760;
const SVG_HEIGHT = 400;
const PLOT = {
  left: 62,
  right: 730,
  top: 24,
  bottom: 346,
};

function MathInline({ tex }: MathInlineProps) {
  return (
    <span
      className="math-inline"
      dangerouslySetInnerHTML={{
        __html: katex.renderToString(tex, {
          displayMode: false,
          throwOnError: false,
        }),
      }}
    />
  );
}

function seededRandom(seed: number) {
  let t = seed;
  return () => {
    t += 0x6d2b79f5;
    let x = t;
    x = Math.imul(x ^ (x >>> 15), x | 1);
    x ^= x + Math.imul(x ^ (x >>> 7), x | 61);
    return ((x ^ (x >>> 14)) >>> 0) / 4294967296;
  };
}

function generateNormals(size: number, seed: number): number[] {
  const random = seededRandom(seed);
  const values: number[] = [];

  while (values.length < size) {
    const u1 = Math.max(Number.EPSILON, random());
    const u2 = random();
    const radius = Math.sqrt(-2 * Math.log(u1));
    const angle = 2 * Math.PI * u2;
    values.push(radius * Math.cos(angle));
    if (values.length < size) values.push(radius * Math.sin(angle));
  }

  return values;
}

function mean(values: number[]): number {
  let total = 0;
  for (const value of values) total += value;
  return total / values.length;
}

function standardize(values: number[]): number[] {
  const xbar = mean(values);
  let ss = 0;
  for (const value of values) {
    const centered = value - xbar;
    ss += centered * centered;
  }
  const sd = Math.sqrt(ss / values.length);
  return values.map((value) => (value - xbar) / sd);
}

function makeBaseDraws(size: number): BaseDraw[] {
  const zx = standardize(generateNormals(size, 19_117));
  const rawU = standardize(generateNormals(size, 82_931));
  let projection = 0;

  for (let i = 0; i < size; i += 1) {
    projection += zx[i] * rawU[i];
  }
  projection /= size;

  const residualU = rawU.map((value, i) => value - projection * zx[i]);
  const u = standardize(residualU);

  return zx.map((value, i) => ({ zx: value, u: u[i] }));
}

function makePopulation(
  baseDraws: BaseDraw[],
  meanHeight: number,
  meanWeight: number,
  sdHeight: number,
  sdWeight: number,
  correlation: number
): Person[] {
  const independentWeightShare = Math.sqrt(Math.max(0, 1 - correlation * correlation));

  return baseDraws.map((draw, id) => {
    const standardizedWeight = correlation * draw.zx + independentWeightShare * draw.u;
    return {
      id,
      height: meanHeight + sdHeight * draw.zx,
      weight: meanWeight + sdWeight * standardizedWeight,
    };
  });
}

function computeMoments(population: Person[]): MomentSummary {
  let totalX = 0;
  let totalY = 0;
  for (const person of population) {
    totalX += person.height;
    totalY += person.weight;
  }

  const meanX = totalX / population.length;
  const meanY = totalY / population.length;
  let varX = 0;
  let covXY = 0;

  for (const person of population) {
    const dx = person.height - meanX;
    const dy = person.weight - meanY;
    varX += dx * dx;
    covXY += dx * dy;
  }

  varX /= population.length;
  covXY /= population.length;

  const slope = covXY / varX;
  const intercept = meanY - slope * meanX;

  return { meanX, meanY, varX, covXY, slope, intercept };
}

function fitOls(sample: Person[]): OlsSummary {
  let totalX = 0;
  let totalY = 0;
  for (const person of sample) {
    totalX += person.height;
    totalY += person.weight;
  }

  const meanX = totalX / sample.length;
  const meanY = totalY / sample.length;
  let centeredXX = 0;
  let centeredXY = 0;

  for (const person of sample) {
    const dx = person.height - meanX;
    centeredXX += dx * dx;
    centeredXY += dx * (person.weight - meanY);
  }

  const slope = centeredXX === 0 ? 0 : centeredXY / centeredXX;
  const intercept = meanY - slope * meanX;
  let residualSumSquares = 0;

  for (const person of sample) {
    const residual = person.weight - (intercept + slope * person.height);
    residualSumSquares += residual * residual;
  }

  const residualDf = sample.length - 2;
  const residualVariance = residualDf > 0 ? residualSumSquares / residualDf : 0;
  const slopeSe = centeredXX === 0 ? 0 : Math.sqrt(residualVariance / centeredXX);
  const interceptSe =
    centeredXX === 0 ? 0 : Math.sqrt(residualVariance * (1 / sample.length + (meanX * meanX) / centeredXX));

  return { intercept, slope, interceptSe, slopeSe };
}

function drawSample(population: Person[], sampleSize: number): Person[] {
  const picked = new Set<number>();
  while (picked.size < sampleSize) {
    picked.add(Math.floor(Math.random() * population.length));
  }
  return Array.from(picked, (idx) => population[idx]);
}

function makePopulationDots(population: Person[]): Person[] {
  const stride = Math.max(1, Math.floor(population.length / MAX_POP_DOTS));
  const dots: Person[] = [];
  for (let i = 0; i < population.length; i += stride) {
    dots.push(population[i]);
  }
  return dots;
}

function paddedDomain(population: Person[]): PlotDomain {
  let xMin = Number.POSITIVE_INFINITY;
  let xMax = Number.NEGATIVE_INFINITY;
  let yMin = Number.POSITIVE_INFINITY;
  let yMax = Number.NEGATIVE_INFINITY;

  for (const person of population) {
    xMin = Math.min(xMin, person.height);
    xMax = Math.max(xMax, person.height);
    yMin = Math.min(yMin, person.weight);
    yMax = Math.max(yMax, person.weight);
  }

  const xPad = Math.max(1, (xMax - xMin) * 0.08);
  const yPad = Math.max(5, (yMax - yMin) * 0.08);

  return {
    xMin: Math.floor(xMin - xPad),
    xMax: Math.ceil(xMax + xPad),
    yMin: Math.floor((yMin - yPad) / 5) * 5,
    yMax: Math.ceil((yMax + yPad) / 5) * 5,
  };
}

function niceStep(rawStep: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const normalized = rawStep / magnitude;
  if (normalized <= 1) return magnitude;
  if (normalized <= 2) return 2 * magnitude;
  if (normalized <= 5) return 5 * magnitude;
  return 10 * magnitude;
}

function makeTicks(min: number, max: number, targetCount: number): number[] {
  const step = niceStep((max - min) / targetCount);
  const first = Math.ceil(min / step) * step;
  const ticks: number[] = [];

  for (let value = first; value <= max + step * 0.5; value += step) {
    ticks.push(Number(value.toFixed(8)));
  }

  return ticks;
}

function formatTick(value: number): string {
  return Math.abs(value) >= 100 || Number.isInteger(value) ? value.toFixed(0) : value.toFixed(1);
}

function formatEstimate(value: number, digits = 2): string {
  return value.toFixed(digits);
}

function RegressionPlot({
  ariaLabel,
  clipId,
  domain,
  meanPoint,
  populationDots,
  populationLine,
  recentLines = [],
  sampleLine = null,
  samplePoints = [],
  showSampleLegend = false,
}: RegressionPlotProps) {
  const plotWidth = PLOT.right - PLOT.left;
  const plotHeight = PLOT.bottom - PLOT.top;
  const xTicks = makeTicks(domain.xMin, domain.xMax, 6);
  const yTicks = makeTicks(domain.yMin, domain.yMax, 5);
  const mapX = (value: number) => PLOT.left + ((value - domain.xMin) / (domain.xMax - domain.xMin)) * plotWidth;
  const mapY = (value: number) => PLOT.bottom - ((value - domain.yMin) / (domain.yMax - domain.yMin)) * plotHeight;
  const lineEndpoints = (line: RegressionLine) => {
    const x1 = domain.xMin;
    const x2 = domain.xMax;
    return {
      x1: mapX(x1),
      y1: mapY(line.intercept + line.slope * x1),
      x2: mapX(x2),
      y2: mapY(line.intercept + line.slope * x2),
    };
  };
  const populationEndpoints = lineEndpoints(populationLine);
  const sampleEndpoints = sampleLine ? lineEndpoints(sampleLine) : null;

  return (
    <svg viewBox={`0 0 ${SVG_WIDTH} ${SVG_HEIGHT}`} className="svg-chart regression-chart" role="img" aria-label={ariaLabel}>
      <defs>
        <clipPath id={clipId}>
          <rect x={PLOT.left} y={PLOT.top} width={plotWidth} height={plotHeight} />
        </clipPath>
      </defs>

      <rect x={PLOT.left} y={PLOT.top} width={plotWidth} height={plotHeight} className="plot-bg" />

      {xTicks.map((tick) => (
        <g key={`x-${tick}`}>
          <line x1={mapX(tick)} x2={mapX(tick)} y1={PLOT.top} y2={PLOT.bottom} className="grid-line" />
          <text x={mapX(tick)} y={PLOT.bottom + 18} textAnchor="middle" className="axis-label">
            {formatTick(tick)}
          </text>
        </g>
      ))}

      {yTicks.map((tick) => (
        <g key={`y-${tick}`}>
          <line x1={PLOT.left} x2={PLOT.right} y1={mapY(tick)} y2={mapY(tick)} className="grid-line" />
          <text x={PLOT.left - 10} y={mapY(tick) + 4} textAnchor="end" className="axis-label">
            {formatTick(tick)}
          </text>
        </g>
      ))}

      <g clipPath={`url(#${clipId})`}>
        {populationDots.map((person) => (
          <circle
            key={`pop-${person.id}`}
            cx={mapX(person.height)}
            cy={mapY(person.weight)}
            r={1.45}
            className="population-dot"
          />
        ))}

        {recentLines.map((line, idx) => {
          const endpoints = lineEndpoints(line);
          return (
            <line
              key={`recent-${idx}-${line.intercept.toFixed(3)}-${line.slope.toFixed(3)}`}
              x1={endpoints.x1}
              y1={endpoints.y1}
              x2={endpoints.x2}
              y2={endpoints.y2}
              className="sample-line-faint"
            />
          );
        })}

        <line
          x1={populationEndpoints.x1}
          y1={populationEndpoints.y1}
          x2={populationEndpoints.x2}
          y2={populationEndpoints.y2}
          className="population-line"
        />

        <line x1={mapX(meanPoint.x)} x2={mapX(meanPoint.x)} y1={PLOT.top} y2={PLOT.bottom} className="mean-guide" />
        <line x1={PLOT.left} x2={PLOT.right} y1={mapY(meanPoint.y)} y2={mapY(meanPoint.y)} className="mean-guide" />

        {samplePoints.map((person) => (
          <circle
            key={`sample-${person.id}`}
            cx={mapX(person.height)}
            cy={mapY(person.weight)}
            r={3}
            className="sample-dot"
          />
        ))}

        {sampleEndpoints && (
          <line
            x1={sampleEndpoints.x1}
            y1={sampleEndpoints.y1}
            x2={sampleEndpoints.x2}
            y2={sampleEndpoints.y2}
            className="sample-line"
          />
        )}

        <circle cx={mapX(meanPoint.x)} cy={mapY(meanPoint.y)} r={5} className="mean-point" />
      </g>

      <line x1={PLOT.left} x2={PLOT.right} y1={PLOT.bottom} y2={PLOT.bottom} className="axis" />
      <line x1={PLOT.left} x2={PLOT.left} y1={PLOT.top} y2={PLOT.bottom} className="axis" />

      <text x={(PLOT.left + PLOT.right) / 2} y={SVG_HEIGHT - 12} textAnchor="middle" className="axis-title">
        Height, X (in)
      </text>
      <text
        x={-((PLOT.top + PLOT.bottom) / 2)}
        y={18}
        textAnchor="middle"
        transform="rotate(-90)"
        className="axis-title"
      >
        Weight, Y (lb)
      </text>

      <g className="legend" transform="translate(512 34)">
        <line x1={0} x2={22} y1={0} y2={0} className="population-line" />
        <text x={30} y={4}>Population line</text>
        {showSampleLegend && (
          <>
            <line x1={0} x2={22} y1={18} y2={18} className="sample-line" />
            <text x={30} y={22}>Sample OLS line</text>
          </>
        )}
        <circle cx={11} cy={showSampleLegend ? 36 : 18} r={4} className="mean-point" />
        <text x={30} y={showSampleLegend ? 40 : 22}>Population mean</text>
      </g>
    </svg>
  );
}

export default function App() {
  const [meanHeight, setMeanHeight] = useState(68);
  const [meanWeight, setMeanWeight] = useState(170);
  const [sdHeight, setSdHeight] = useState(3);
  const [sdWeight, setSdWeight] = useState(25);
  const [correlation, setCorrelation] = useState(0.55);
  const [sampleSize, setSampleSize] = useState(100);
  const [speed, setSpeed] = useState<Speed>('medium');
  const [isRunning, setIsRunning] = useState(false);
  const [currentSample, setCurrentSample] = useState<Person[]>([]);
  const [sampleLine, setSampleLine] = useState<OlsSummary | null>(null);
  const [recentLines, setRecentLines] = useState<RegressionLine[]>([]);

  const baseDraws = useMemo(() => makeBaseDraws(NPOP), []);
  const population = useMemo(
    () => makePopulation(baseDraws, meanHeight, meanWeight, sdHeight, sdWeight, correlation),
    [baseDraws, correlation, meanHeight, meanWeight, sdHeight, sdWeight]
  );
  const moments = useMemo(() => computeMoments(population), [population]);
  const populationDots = useMemo(() => makePopulationDots(population), [population]);
  const domain = useMemo(() => paddedDomain(population), [population]);

  useEffect(() => {
    setIsRunning(false);
    setCurrentSample([]);
    setSampleLine(null);
    setRecentLines([]);
  }, [population, sampleSize]);

  const runOneSample = useCallback(() => {
    const sample = drawSample(population, sampleSize);
    const line = fitOls(sample);
    setCurrentSample(sample);
    setSampleLine(line);
    setRecentLines((prev) => [...prev, line].slice(-MAX_RECENT_LINES));
  }, [population, sampleSize]);

  useEffect(() => {
    if (!isRunning) return undefined;
    const id = window.setInterval(() => {
      runOneSample();
    }, SPEED_MS[speed]);
    return () => window.clearInterval(id);
  }, [isRunning, runOneSample, speed]);

  const handleStep = () => {
    setIsRunning(false);
    runOneSample();
  };

  const handleStart = () => setIsRunning(true);

  const handlePause = () => setIsRunning(false);

  const clearSamples = () => {
    setIsRunning(false);
    setCurrentSample([]);
    setSampleLine(null);
    setRecentLines([]);
  };

  const sampleSlope = sampleLine ? formatEstimate(sampleLine.slope) : '—';
  const sampleIntercept = sampleLine ? formatEstimate(sampleLine.intercept, 1) : '—';
  const sampleSlopeSe = sampleLine ? formatEstimate(sampleLine.slopeSe) : '—';
  const sampleInterceptSe = sampleLine ? formatEstimate(sampleLine.interceptSe, 1) : '—';

  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>Population and Sample Regression Explorer</h1>
          <p className="subtitle">Econ 1117 – Yale University</p>
        </div>
        <img className="yale-logo" src={import.meta.env.BASE_URL + 'yale_logo.png'} alt="Yale University logo" />
      </header>

      <div className="description">
        <p>
          This simulator distinguishes the population linear regression line from the OLS regression line estimated in
          a random sample.
        </p>
        <p>
          The example uses height (in) and weight (lb). Move the sliders to change the population cloud, then draw
          samples to see how OLS estimates vary.
        </p>
      </div>

      <section className="panel">
        <h2>Panel 1: Population Regression</h2>
        <div className="panel-subtitle">
          The population regression line summarizes the linear relationship between <MathInline tex={'Y'} /> and <MathInline tex={'X'} /> in the  population. In this panel, you can modify features of the distribution of <MathInline tex={'(Y,X)'} /> and see how this impacts the population regression line.
        </div>

        <div className="controls panel-controls">
          <label>
            <span>
              Mean height <MathInline tex={'E[X]'} />
            </span>
            <input
              type="range"
              min={62}
              max={74}
              step={0.5}
              value={meanHeight}
              onChange={(event) => setMeanHeight(Number(event.target.value))}
            />
            <span className="control-readout">{meanHeight.toFixed(1)} in</span>
          </label>

          <label>
            <span>
              Mean weight <MathInline tex={'E[Y]'} />
            </span>
            <input
              type="range"
              min={120}
              max={220}
              step={5}
              value={meanWeight}
              onChange={(event) => setMeanWeight(Number(event.target.value))}
            />
            <span className="control-readout">{meanWeight.toFixed(0)} lb</span>
          </label>

          <label>
            <span>
              SD of height <MathInline tex={'\\sigma_X'} />
            </span>
            <input
              type="range"
              min={1.5}
              max={5.5}
              step={0.1}
              value={sdHeight}
              onChange={(event) => setSdHeight(Number(event.target.value))}
            />
            <span className="control-readout">{sdHeight.toFixed(1)} in</span>
          </label>

          <label>
            <span>
              SD of weight <MathInline tex={'\\sigma_Y'} />
            </span>
            <input
              type="range"
              min={10}
              max={45}
              step={1}
              value={sdWeight}
              onChange={(event) => setSdWeight(Number(event.target.value))}
            />
            <span className="control-readout">{sdWeight.toFixed(0)} lb</span>
          </label>

          <label>
            <span>
              Correlation between height and weight <MathInline tex={'\\rho'} />
            </span>
            <input
              type="range"
              min={0}
              max={0.9}
              step={0.05}
              value={correlation}
              onChange={(event) => setCorrelation(Number(event.target.value))}
            />
            <span className="control-readout">{correlation.toFixed(2)}</span>
          </label>
        </div>

        <div className="stats regression-stats">
          <div>
            <span className="stat-label">Mean height</span>
            <strong>
              <MathInline tex={'E[X]'} /> = {formatEstimate(moments.meanX)} in
            </strong>
          </div>
          <div>
            <span className="stat-label">Mean weight</span>
            <strong>
              <MathInline tex={'E[Y]'} /> = {formatEstimate(moments.meanY, 1)} lb
            </strong>
          </div>
          <div>
            <span className="stat-label">Variance of height</span>
            <strong>
              <MathInline tex={'\\operatorname{Var}(X)'} /> = {formatEstimate(moments.varX)}
            </strong>
          </div>
          <div>
            <span className="stat-label">Covariance of height and weight</span>
            <strong>
              <MathInline tex={'\\operatorname{Cov}(X,Y)'} /> = {formatEstimate(moments.covXY)}
            </strong>
          </div>
          <div>
            <span className="stat-label">Population slope</span>
            <strong className="population-estimate">
              <MathInline tex={'\\beta_1 = \\frac{\\operatorname{Cov}(X,Y)}{\\operatorname{Var}(X)}'} /> ={' '}
              {formatEstimate(moments.slope)}
            </strong>
          </div>
          <div>
            <span className="stat-label">Population intercept</span>
            <strong className="population-estimate">
              <MathInline tex={'\\beta_0 = E[Y] - \\beta_1 E[X]'} /> = {formatEstimate(moments.intercept, 1)}
            </strong>
          </div>
        </div>

        <RegressionPlot
          ariaLabel="Population height and weight scatterplot with population regression line"
          clipId="population-plot-clip"
          domain={domain}
          meanPoint={{ x: moments.meanX, y: moments.meanY }}
          populationDots={populationDots}
          populationLine={moments}
        />

        <p className="plot-note">
          The population regression line is the line of best fit for the joint distribution of height and weight. Its slope is determined by the covariance between height and weight relative to the variance of height, and its intercept ensures that the line passes through the population mean point. The line itself is a fixed feature of the population; we use sample data to estimate it.
        </p>
      </section>

      <section className="panel">
        <h2>Panel 2: Sampling Variation in Regression</h2>
        <div className="panel-subtitle">
          In real data, we usually observe a sample. Each random sample leads to its own OLS estimate of the population regression line. In this panel, you can draw different random samples after choosing sample size, and see how the estimated regression line compares to the population one.
        </div>

        <div className="controls sample-controls">
          <label>
            Sample size <MathInline tex={'N'} />
            <input
              type="range"
              min={10}
              max={1000}
              step={10}
              value={sampleSize}
              onChange={(event) => setSampleSize(Number(event.target.value))}
            />
            <span className="control-readout">N = {sampleSize}</span>
          </label>

          <label>
            Speed
            <select value={speed} onChange={(event) => setSpeed(event.target.value as Speed)}>
              <option value="slow">Slow</option>
              <option value="medium">Medium</option>
              <option value="fast">Fast</option>
            </select>
          </label>

          <div className="buttons sample-buttons">
            <button type="button" onClick={handleStart} disabled={isRunning}>
              Start
            </button>
            <button type="button" onClick={handlePause} disabled={!isRunning}>
              Pause
            </button>
            <button type="button" onClick={handleStep}>
              Draw
            </button>
            <button type="button" onClick={clearSamples}>
              Reset
            </button>
          </div>
        </div>

        <div className="stats regression-stats ols-stats">
          <div>
            <span className="stat-label">Population slope</span>
            <strong className="population-estimate">
              <MathInline tex={'\\beta_1'} /> = {formatEstimate(moments.slope)}
            </strong>
          </div>
          <div>
            <span className="stat-label">Sample slope</span>
            <strong className="sample-estimate">
              <MathInline tex={'\\hat{\\beta}_1'} /> = {sampleSlope}
            </strong>
          </div>
          <div>
            <span className="stat-label">SE of sample slope</span>
            <strong className="sample-estimate">
              <MathInline tex={'\\operatorname{SE}(\\hat{\\beta}_1)'} /> = {sampleSlopeSe}
            </strong>
          </div>
          <div>
            <span className="stat-label">Population intercept</span>
            <strong className="population-estimate">
              <MathInline tex={'\\beta_0'} /> = {formatEstimate(moments.intercept, 1)}
            </strong>
          </div>
          <div>
            <span className="stat-label">Sample intercept</span>
            <strong className="sample-estimate">
              <MathInline tex={'\\hat{\\beta}_0'} /> = {sampleIntercept}
            </strong>
          </div>
          <div>
            <span className="stat-label">SE of sample intercept</span>
            <strong className="sample-estimate">
              <MathInline tex={'\\operatorname{SE}(\\hat{\\beta}_0)'} /> = {sampleInterceptSe}
            </strong>
          </div>
        </div>

        <RegressionPlot
          ariaLabel="Population scatterplot with highlighted sample points and sample OLS regression line"
          clipId="sample-plot-clip"
          domain={domain}
          meanPoint={{ x: moments.meanX, y: moments.meanY }}
          populationDots={populationDots}
          populationLine={moments}
          recentLines={recentLines}
          sampleLine={sampleLine}
          samplePoints={currentSample}
          showSampleLegend
        />

        <p className="plot-note">
          The orange line is fixed because it belongs to the population. The blue OLS line changes when the random sample
          changes.
        </p>
      </section>

      <footer className="footer-credit">
        Interactive visualization by{' '}
        <a
          href="https://www.jarellanobover.com/"
          target="_blank"
          rel="noopener noreferrer"
          aria-label="Open Jaime Arellano-Bover’s website in a new tab"
        >
          Jaime Arellano-Bover
        </a>
      </footer>
    </div>
  );
}
