asset_candidates = [
# from S&P500
('A', 'Agilent Technologies Inc'),
('AAL', 'American Airlines Group Inc'),
('AAP', 'Advance Auto Parts, Inc.'),
('AAPL', 'Apple Inc.'),
('ABBV', 'AbbVie Inc'),
('ABC', 'AmerisourceBergen Corp.'),
('ABMD', 'ABIOMED, Inc.'),
('ABT', 'Abbott Laboratories'),
('ACN', 'Accenture Plc'),
('ADBE', 'Adobe Inc'),
('ADI', 'ADI'),
('ADM', 'ADM'),
('ADP', 'ADP'),
('ADS', 'ADS'),
('ADSK', 'ADSK'),
('AEE', 'AEE'),
('AEP', 'AEP'),
('AES', 'AES'),
('AFL', 'AFL'),
('AGN', 'AGN'),
('AIG', 'AIG'),
('AIV', 'AIV'),
('AIZ', 'AIZ'),
('AJG', 'AJG'),
('AKAM', 'AKAM'),
('ALB', 'ALB'),
('ALGN', 'ALGN'),
('ALK', 'ALK'),
('ALL', 'ALL'),
('ALLE', 'ALLE'),
('ALXN', 'ALXN'),
('AMAT', 'AMAT'),
('AMCR', 'AMCR'),
('AMD', 'AMD'),
('AME', 'AME'),
('AMGN', 'AMGN'),
('AMP', 'AMP'),
('AMT', 'AMT'),
('AMZN', 'AMZN'),
('ANET', 'ANET'),
('ANSS', 'ANSS'),
('ANTM', 'ANTM'),
('AON', 'AON'),
('AOS', 'AOS'),
('APA', 'APA'),
('APD', 'APD'),
('APH', 'APH'),
('APTV', 'APTV'),
('ARE', 'ARE'),
('ARNC', 'ARNC'),
('ATO', 'ATO'),
('ATVI', 'ATVI'),
('AVB', 'AVB'),
('AVGO', 'AVGO'),
('AVY', 'AVY'),
('AWK', 'AWK'),
('AXP', 'AXP'),
('AZO', 'AZO'),
('BA', 'BA'),
('BAC', 'BAC'),
('BAX', 'BAX'),
('BBY', 'BBY'),
('BDX', 'BDX'),
('BEN', 'BEN'),
('BF-B', 'BF-B'),
('BIIB', 'BIIB'),
('BK', 'BK'),
('BKNG', 'BKNG'),
('BKR', 'BKR'),
('BLK', 'BLK'),
('BLL', 'BLL'),
('BMY', 'BMY'),
('BR', 'BR'),
('BRK-B', 'BRK-B'),
('BSX', 'BSX'),
('BWA', 'BWA'),
('BXP', 'BXP'),
('C', 'C'),
('CAG', 'CAG'),
('CAH', 'CAH'),
('CAT', 'CAT'),
('CB', 'CB'),
('CBOE', 'CBOE'),
('CBRE', 'CBRE'),
('CCI', 'CCI'),
('CCL', 'CCL'),
('CDNS', 'CDNS'),
('CDW', 'CDW'),
('CE', 'CE'),
('CERN', 'CERN'),
('CF', 'CF'),
('CFG', 'CFG'),
('CHD', 'CHD'),
('CHRW', 'CHRW'),
('CHTR', 'CHTR'),
('CI', 'CI'),
('CINF', 'CINF'),
('CL', 'CL'),
('CLX', 'CLX'),
('CMA', 'CMA'),
('CMCSA', 'CMCSA'),
('CME', 'CME'),
('CMG', 'CMG'),
('CMI', 'CMI'),
('CMS', 'CMS'),
('CNC', 'CNC'),
('CNP', 'CNP'),
('COF', 'COF'),
('COG', 'COG'),
('COO', 'COO'),
('COP', 'COP'),
('COST', 'COST'),
('COTY', 'COTY'),
('CPB', 'CPB'),
('CPRI', 'CPRI'),
('CPRT', 'CPRT'),
('CRM', 'CRM'),
('CSCO', 'CSCO'),
('CSX', 'CSX'),
('CTAS', 'CTAS'),
('CTL', 'CTL'),
('CTSH', 'CTSH'),
('CTVA', 'CTVA'),
('CTXS', 'CTXS'),
('CVS', 'CVS'),
('CVX', 'CVX'),
('CXO', 'CXO'),
('D', 'D'),
('DAL', 'DAL'),
('DD', 'DD'),
('DE', 'DE'),
('DFS', 'DFS'),
('DG', 'DG'),
('DGX', 'DGX'),
('DHI', 'DHI'),
('DHR', 'DHR'),
('DIS', 'DIS'),
('DISCA', 'DISCA'),
('DISCK', 'DISCK'),
('DISH', 'DISH'),
('DLR', 'DLR'),
('DLTR', 'DLTR'),
('DOV', 'DOV'),
('DOW', 'DOW'),
('DRE', 'DRE'),
('DRI', 'DRI'),
('DTE', 'DTE'),
('DUK', 'DUK'),
('DVA', 'DVA'),
('DVN', 'DVN'),
('DXC', 'DXC'),
('EA', 'EA'),
('EBAY', 'EBAY'),
('ECL', 'ECL'),
('ED', 'ED'),
('EFX', 'EFX'),
('EIX', 'EIX'),
('EL', 'EL'),
('EMN', 'EMN'),
('EMR', 'EMR'),
('EOG', 'EOG'),
('EQIX', 'EQIX'),
('EQR', 'EQR'),
('ES', 'ES'),
('ESS', 'ESS'),
('ETFC', 'ETFC'),
('ETN', 'ETN'),
('ETR', 'ETR'),
('EVRG', 'EVRG'),
('EW', 'EW'),
('EXC', 'EXC'),
('EXPD', 'EXPD'),
('EXPE', 'EXPE'),
('EXR', 'EXR'),
('F', 'F'),
('FANG', 'FANG'),
('FAST', 'FAST'),
('FB', 'FB'),
('FBHS', 'FBHS'),
('FCX', 'FCX'),
('FDX', 'FDX'),
('FE', 'FE'),
('FFIV', 'FFIV'),
('FIS', 'FIS'),
('FISV', 'FISV'),
('FITB', 'FITB'),
('FLIR', 'FLIR'),
('FLS', 'FLS'),
('FLT', 'FLT'),
('FMC', 'FMC'),
('FOX', 'FOX'),
('FOXA', 'FOXA'),
('FRC', 'FRC'),
('FRT', 'FRT'),
('FTI', 'FTI'),
('FTNT', 'FTNT'),
('FTV', 'FTV'),
('GD', 'GD'),
('GE', 'GE'),
('GILD', 'GILD'),
('GIS', 'GIS'),
('GL', 'GL'),
('GLW', 'GLW'),
('GM', 'GM'),
('GOOG', 'GOOG'),
('GOOGL', 'GOOGL'),
('GPC', 'GPC'),
('GPN', 'GPN'),
('GPS', 'GPS'),
('GRMN', 'GRMN'),
('GS', 'GS'),
('GWW', 'GWW'),
('HAL', 'HAL'),
('HAS', 'HAS'),
('HBAN', 'HBAN'),
('HBI', 'HBI'),
('HCA', 'HCA'),
('HD', 'HD'),
('HES', 'HES'),
('HFC', 'HFC'),
('HIG', 'HIG'),
('HII', 'HII'),
('HLT', 'HLT'),
('HOG', 'HOG'),
('HOLX', 'HOLX'),
('HON', 'HON'),
('HP', 'HP'),
('HPE', 'HPE'),
('HPQ', 'HPQ'),
('HRB', 'HRB'),
('HRL', 'HRL'),
('HSIC', 'HSIC'),
('HST', 'HST'),
('HSY', 'HSY'),
('HUM', 'HUM'),
('IBM', 'IBM'),
('ICE', 'ICE'),
('IDXX', 'IDXX'),
('IEX', 'IEX'),
('IFF', 'IFF'),
('ILMN', 'ILMN'),
('INCY', 'INCY'),
('INFO', 'INFO'),
('INTC', 'INTC'),
('INTU', 'INTU'),
('IP', 'IP'),
('IPG', 'IPG'),
('IPGP', 'IPGP'),
('IQV', 'IQV'),
('IR', 'IR'),
('IRM', 'IRM'),
('ISRG', 'ISRG'),
('IT', 'IT'),
('ITW', 'ITW'),
('IVZ', 'IVZ'),
('J', 'J'),
('JBHT', 'JBHT'),
('JCI', 'JCI'),
('JKHY', 'JKHY'),
('JNJ', 'JNJ'),
('JNPR', 'JNPR'),
('JPM', 'JPM'),
('JWN', 'JWN'),
('K', 'K'),
('KEY', 'KEY'),
('KEYS', 'KEYS'),
('KHC', 'KHC'),
('KIM', 'KIM'),
('KLAC', 'KLAC'),
('KMB', 'KMB'),
('KMI', 'KMI'),
('KMX', 'KMX'),
('KO', 'KO'),
('KR', 'KR'),
('KSS', 'KSS'),
('KSU', 'KSU'),
('L', 'L'),
('LB', 'LB'),
('LDOS', 'LDOS'),
('LEG', 'LEG'),
('LEN', 'LEN'),
('LH', 'LH'),
('LHX', 'LHX'),
('LIN', 'LIN'),
('LKQ', 'LKQ'),
('LLY', 'LLY'),
('LMT', 'LMT'),
('LNC', 'LNC'),
('LNT', 'LNT'),
('LOW', 'LOW'),
('LRCX', 'LRCX'),
('LUV', 'LUV'),
('LVS', 'LVS'),
('LW', 'LW'),
('LYB', 'LYB'),
('LYV', 'LYV'),
('M', 'M'),
('MA', 'MA'),
('MAA', 'MAA'),
('MAR', 'MAR'),
('MAS', 'MAS'),
('MCD', 'MCD'),
('MCHP', 'MCHP'),
('MCK', 'MCK'),
('MCO', 'MCO'),
('MDLZ', 'MDLZ'),
('MDT', 'MDT'),
('MET', 'MET'),
('MGM', 'MGM'),
('MHK', 'MHK'),
('MKC', 'MKC'),
('MKTX', 'MKTX'),
('MLM', 'MLM'),
('MMC', 'MMC'),
('MMM', 'MMM'),
('MNST', 'MNST'),
('MO', 'MO'),
('MOS', 'MOS'),
('MPC', 'MPC'),
('MRK', 'MRK'),
('MRO', 'MRO'),
('MS', 'MS'),
('MSCI', 'MSCI'),
('MSFT', 'MSFT'),
('MSI', 'MSI'),
('MTB', 'MTB'),
('MTD', 'MTD'),
('MU', 'MU'),
('MXIM', 'MXIM'),
('MYL', 'MYL'),
('NBL', 'NBL'),
('NCLH', 'NCLH'),
('NDAQ', 'NDAQ'),
('NEE', 'NEE'),
('NEM', 'NEM'),
('NFLX', 'NFLX'),
('NI', 'NI'),
('NKE', 'NKE'),
('NLOK', 'NLOK'),
('NLSN', 'NLSN'),
('NOC', 'NOC'),
('NOV', 'NOV'),
('NOW', 'NOW'),
('NRG', 'NRG'),
('NSC', 'NSC'),
('NTAP', 'NTAP'),
('NTRS', 'NTRS'),
('NUE', 'NUE'),
('NVDA', 'NVDA'),
('NVR', 'NVR'),
('NWL', 'NWL'),
('NWS', 'NWS'),
('NWSA', 'NWSA'),
('O', 'O'),
('ODFL', 'ODFL'),
('OKE', 'OKE'),
('OMC', 'OMC'),
('ORCL', 'ORCL'),
('ORLY', 'ORLY'),
('OXY', 'OXY'),
('PAYC', 'PAYC'),
('PAYX', 'PAYX'),
('PBCT', 'PBCT'),
('PCAR', 'PCAR'),
('PEAK', 'PEAK'),
('PEG', 'PEG'),
('PEP', 'PEP'),
('PFE', 'PFE'),
('PFG', 'PFG'),
('PG', 'PG'),
('PGR', 'PGR'),
('PH', 'PH'),
('PHM', 'PHM'),
('PKG', 'PKG'),
('PKI', 'PKI'),
('PLD', 'PLD'),
('PM', 'PM'),
('PNC', 'PNC'),
('PNR', 'PNR'),
('PNW', 'PNW'),
('PPG', 'PPG'),
('PPL', 'PPL'),
('PRGO', 'PRGO'),
('PRU', 'PRU'),
('PSA', 'PSA'),
('PSX', 'PSX'),
('PVH', 'PVH'),
('PWR', 'PWR'),
('PXD', 'PXD'),
('PYPL', 'PYPL'),
('QCOM', 'QCOM'),
('QRVO', 'QRVO'),
('RCL', 'RCL'),
('RE', 'RE'),
('REG', 'REG'),
('REGN', 'REGN'),
('RF', 'RF'),
('RHI', 'RHI'),
('RJF', 'RJF'),
('RL', 'RL'),
('RMD', 'RMD'),
('ROK', 'ROK'),
('ROL', 'ROL'),
('ROP', 'ROP'),
('ROST', 'ROST'),
('RSG', 'RSG'),
('RTX', 'RTX'),
('SBAC', 'SBAC'),
('SBUX', 'SBUX'),
('SCHW', 'SCHW'),
('SEE', 'SEE'),
('SHW', 'SHW'),
('SIVB', 'SIVB'),
('SJM', 'SJM'),
('SLB', 'SLB'),
('SLG', 'SLG'),
('SNA', 'SNA'),
('SNPS', 'SNPS'),
('SO', 'SO'),
('SPG', 'SPG'),
('SPGI', 'SPGI'),
('SRE', 'SRE'),
('STE', 'STE'),
('STT', 'STT'),
('STX', 'STX'),
('STZ', 'STZ'),
('SWK', 'SWK'),
('SWKS', 'SWKS'),
('SYF', 'SYF'),
('SYK', 'SYK'),
('SYY', 'SYY'),
('T', 'T'),
('TAP', 'TAP'),
('TDG', 'TDG'),
('TEL', 'TEL'),
('TFC', 'TFC'),
('TFX', 'TFX'),
('TGT', 'TGT'),
('TIF', 'TIF'),
('TJX', 'TJX'),
('TMO', 'TMO'),
('TMUS', 'TMUS'),
('TPR', 'TPR'),
('TROW', 'TROW'),
('TRV', 'TRV'),
('TSCO', 'TSCO'),
('TSN', 'TSN'),
('TT', 'TT'),
('TTWO', 'TTWO'),
('TWTR', 'TWTR'),
('TXN', 'TXN'),
('TXT', 'TXT'),
('UA', 'UA'),
('UAA', 'UAA'),
('UAL', 'UAL'),
('UDR', 'UDR'),
('UHS', 'UHS'),
('ULTA', 'ULTA'),
('UNH', 'UNH'),
('UNM', 'UNM'),
('UNP', 'UNP'),
('UPS', 'UPS'),
('URI', 'URI'),
('USB', 'USB'),
('V', 'V'),
('VAR', 'VAR'),
('VFC', 'VFC'),
('VIAC', 'VIAC'),
('VLO', 'VLO'),
('VMC', 'VMC'),
('VNO', 'VNO'),
('VRSK', 'VRSK'),
('VRSN', 'VRSN'),
('VRTX', 'VRTX'),
('VTR', 'VTR'),
('VZ', 'VZ'),
('WAB', 'WAB'),
('WAT', 'WAT'),
('WBA', 'WBA'),
('WDC', 'WDC'),
('WEC', 'WEC'),
('WELL', 'WELL'),
('WFC', 'WFC'),
('WHR', 'WHR'),
('WLTW', 'WLTW'),
('WM', 'WM'),
('WMB', 'WMB'),
('WMT', 'WMT'),
('WRB', 'WRB'),
('WRK', 'WRK'),
('WU', 'WU'),
('WY', 'WY'),
('WYNN', 'WYNN'),
('XEL', 'XEL'),
('XLNX', 'XLNX'),
('XOM', 'XOM'),
('XRAY', 'XRAY'),
('XRX', 'XRX'),
('XYL', 'XYL'),
('YUM', 'YUM'),
('ZBH', 'ZBH'),
('ZBRA', 'ZBRA'),
('ZION', 'ZION'),
('ZTS', 'ZTS'),
# from TradingValley, no data yet
('BIV', 'Vanguard Intermediate Term Bond Index Fund'),
('LQD', 'iShares iBoxx $ Investment Grade Corporate Bond ETF'),
('MUB', 'iShares National Muni Bond ETF'),
('TLT', 'iShares 20+ Year Treasury Bond ETF'),
('VB', 'Vanguard Small-Cap Index Fund'),
('VNQ', 'Vanguard REIT Index Fund'),
('VOO', 'Vanguard S&P 500 Index Fund'),
('VEA', 'Vanguard FTSE Developed Markets ETF'),
('VWO', 'Vanguard FTSE Emerging Markets ETF'),
('IAU', 'iShares Gold Trust')
]