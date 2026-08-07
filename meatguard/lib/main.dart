import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

// Entry point of the MeatGuard application
void main() {
  WidgetsFlutterBinding.ensureInitialized();

  SystemChrome.setSystemUIOverlayStyle(
    const SystemUiOverlayStyle(
      statusBarColor: Colors.transparent,
      statusBarIconBrightness: Brightness.dark,
      statusBarBrightness: Brightness.light,
      systemNavigationBarColor: Colors.white,
      systemNavigationBarIconBrightness: Brightness.dark,
    ),
  );

  runApp(const MeatGuardApp());
}

class MeatGuardApp extends StatelessWidget {
  const MeatGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MeatGuard',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        primarySwatch: Colors.red,
        scaffoldBackgroundColor: const Color(0xFFF9FAFB),
        fontFamily: 'Roboto',
      ),
      home: const HomeScreen(),
      // Define named routes for seamless navigation across all 4 screens
      routes: {
        '/home': (context) => const HomeScreen(),
        '/scan': (context) => const ScanScreen(),
        '/reports': (context) => const ReportsScreen(),
        '/history': (context) => const HistoryScreen(),
        '/settings': (context) => const SettingsScreen(),
      },
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  int _currentIndex = 0;

  void _onBottomNavTapped(int index) {
    setState(() {
      _currentIndex = index;
    });

    switch (index) {
      case 0:
      // Already on Home
        break;
      case 1:
        Navigator.pushReplacementNamed(context, '/scan');
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/reports');
        break;
      case 3:
        Navigator.pushReplacementNamed(context, '/history');
        break;
      case 4:
        Navigator.pushReplacementNamed(context, '/settings');
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
        statusBarBrightness: Brightness.light,
      ),
      child: Scaffold(
        key: _scaffoldKey,
        backgroundColor: const Color(0xFFF8F9FA),
        drawer: _buildToggleMenuDrawer(context),
        body: SafeArea(
          child: Column(
            children: [
              _buildTopAppBar(context),

              Expanded(
                child: SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 8.0),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      _buildBannerCard(context),
                      const SizedBox(height: 20),
                      _buildFeatureGrid(context),
                      const SizedBox(height: 20),
                      _buildModelStatusCard(context),
                      const SizedBox(height: 30),
                    ],
                  ),
                ),
              ),

              _buildBottomNavigationBar(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildTopAppBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16.0, vertical: 12.0),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.03),
            blurRadius: 4,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          IconButton(
            icon: const Icon(Icons.menu_rounded, color: Colors.black87, size: 28),
            onPressed: () {
              _scaffoldKey.currentState?.openDrawer();
            },
          ),

          // Logo container: displaying ONLY the imported logo image (icon + name included in asset)
          Expanded(
            child: Center(
              child: Image.asset(
                'lib/assets/images/logo.png',
                height: 36,
                fit: BoxFit.contain,
                errorBuilder: (context, error, stackTrace) {
                  return Row(
                    mainAxisSize: MainAxisSize.min,
                    children: const [
                      Icon(Icons.shield_rounded, color: Color(0xFFE53935), size: 24),
                      SizedBox(width: 6),
                      Text(
                        'MeatGuard',
                        style: TextStyle(
                          fontSize: 20,
                          fontWeight: FontWeight.bold,
                          color: Color(0xFF2D3748),
                        ),
                      ),
                    ],
                  );
                },
              ),
            ),
          ),

          Stack(
            children: [
              IconButton(
                icon: const Icon(Icons.notifications_outlined, color: Colors.black87, size: 26),
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('No new notifications')),
                  );
                },
              ),
              Positioned(
                right: 8,
                top: 8,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: const BoxDecoration(
                    color: Color(0xFFE53935),
                    shape: BoxShape.circle,
                  ),
                  child: const Text(
                    '2',
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 10,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildToggleMenuDrawer(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                DrawerHeader(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Color(0xFFE53935), Color(0xFFB71C1C)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Image.asset(
                          'lib/assets/images/logo.png',
                          height: 48,
                          errorBuilder: (context, error, stackTrace) {
                            return const Icon(Icons.shield_rounded, color: Color(0xFFE53935), size: 40);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.history_rounded, color: Color(0xFFE53935)),
                  title: const Text('Scan history'),
                  onTap: () {
                    Navigator.pop(context);
                    Navigator.pushReplacementNamed(context, '/history');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.memory_rounded, color: Color(0xFFE53935)),
                  title: const Text('Model Center (v1.2.3)'),
                  subtitle: const Text('TFLite Model Active'),
                  onTap: () {
                    Navigator.pop(context);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Model Center: MeatScan v1.2.3 loaded successfully.')),
                    );
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.settings_rounded, color: Color(0xFFE53935)),
                  title: const Text('Settings'),
                  onTap: () {
                    Navigator.pop(context);
                    Navigator.pushReplacementNamed(context, '/settings');
                  },
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.help_outline_rounded, color: Colors.grey),
                  title: const Text('Support & Help Center'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'Support & Help Center', 'Contact support at support@meatguard.ai or visit our documentation portal.');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.info_outline_rounded, color: Colors.grey),
                  title: const Text('About MeatGuard'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'About MeatGuard', 'MeatGuard is an advanced AI-driven beef freshness detection app built for reliable field inspection using TFLite.');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.bug_report_outlined, color: Colors.grey),
                  title: const Text('Feedback & Bug Report'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'Feedback & Bug Report', 'Thank you for helping us improve! Send bug reports to feedback@meatguard.ai');
                  },
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              children: [
                const Text(
                  'MeatGuard AI',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Version 1.2.3 (Production)',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBannerCard(BuildContext context) {
    return Container(
      width: double.infinity,
      height: 210,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        image: DecorationImage(
          image: const AssetImage('lib/assets/images/banner.png'),
          fit: BoxFit.cover,
          onError: (exception, stackTrace) {},
        ),
        color: const Color(0xFF4A1512),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.15),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Stack(
        children: [
          Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              gradient: LinearGradient(
                colors: [Colors.black.withOpacity(0.8), Colors.black.withOpacity(0.3)],
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
              ),
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: const Color(0xFFE53935),
                    borderRadius: BorderRadius.circular(20),
                  ),
                  child: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: const [
                      Icon(Icons.auto_awesome, color: Colors.white, size: 12),
                      SizedBox(width: 4),
                      Text(
                        'AI POWERED',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                          letterSpacing: 0.5,
                        ),
                      ),
                    ],
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Detect Freshness.\nProtect Health.',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 20,
                    fontWeight: FontWeight.bold,
                    height: 1.2,
                  ),
                ),
                const SizedBox(height: 6),
                const Text(
                  'Instantly scan meat and get\nreliable AI-powered results.',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 12),
                InkWell(
                  onTap: () => Navigator.pushReplacementNamed(context, '/scan'),
                  borderRadius: BorderRadius.circular(12),
                  child: Container(
                    padding: const EdgeInsets.symmetric(horizontal: 14, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white,
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Row(
                      mainAxisSize: MainAxisSize.min,
                      children: const [
                        Icon(Icons.camera_alt_outlined, color: Color(0xFF2D3748), size: 16),
                        SizedBox(width: 6),
                        Text(
                          'SCAN MEAT NOW',
                          style: TextStyle(
                            color: Color(0xFF2D3748),
                            fontSize: 12,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                        SizedBox(width: 4),
                        Icon(Icons.chevron_right, color: Color(0xFF2D3748), size: 16),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildFeatureGrid(BuildContext context) {
    return GridView.count(
      crossAxisCount: 2,
      crossAxisSpacing: 16,
      mainAxisSpacing: 16,
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      childAspectRatio: 0.95,
      children: [
        _buildFeatureCard(
          context: context,
          title: 'Scan Meat',
          subtitle: 'Scan a meat sample to check if it is fresh or spoiled',
          icon: Icons.qr_code_scanner_rounded,
          iconColor: const Color(0xFFE53935),
          iconBgColor: const Color(0xFFFFECEE),
          route: '/scan',
        ),
        _buildFeatureCard(
          context: context,
          title: 'Scan History',
          subtitle: 'View your previous scans and results',
          icon: Icons.access_time_rounded,
          iconColor: const Color(0xFF1E88E5),
          iconBgColor: const Color(0xFFE3F2FD),
          route: '/history',
        ),
        _buildFeatureCard(
          context: context,
          title: 'Reports',
          subtitle: 'See analytics and insights about your scans',
          icon: Icons.bar_chart_rounded,
          iconColor: const Color(0xFF43A047),
          iconBgColor: const Color(0xFFE8F5E9),
          route: '/reports',
        ),
        _buildFeatureCard(
          context: context,
          title: 'Settings',
          subtitle: 'Configure app preferences and options',
          icon: Icons.settings_rounded,
          iconColor: const Color(0xFF8E24AA),
          iconBgColor: const Color(0xFFF3E5F5),
          route: '/settings',
        ),
      ],
    );
  }

  Widget _buildFeatureCard({
    required BuildContext context,
    required String title,
    required String subtitle,
    required IconData icon,
    required Color iconColor,
    required Color iconBgColor,
    required String route,
  }) {
    return InkWell(
      onTap: () => Navigator.pushReplacementNamed(context, route),
      borderRadius: BorderRadius.circular(20),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.grey.shade200, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.02),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: iconBgColor,
                    borderRadius: BorderRadius.circular(14),
                  ),
                  child: Icon(icon, color: iconColor, size: 24),
                ),
                Container(
                  padding: const EdgeInsets.all(4),
                  decoration: BoxDecoration(
                    color: Colors.grey.shade100,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.chevron_right, color: Colors.grey.shade600, size: 18),
                ),
              ],
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(
                    fontSize: 16,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2D3748),
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  subtitle,
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                  style: TextStyle(
                    fontSize: 11,
                    color: Colors.grey[600],
                    height: 1.3,
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelStatusCard(BuildContext context) {
    return InkWell(
      onTap: () {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Model Center: MeatScan v1.2.3 (TFLite Active)')),
        );
      },
      borderRadius: BorderRadius.circular(20),
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          border: Border.all(color: Colors.grey.shade200, width: 1),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.02),
              blurRadius: 8,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: const Color(0xFFE8F5E9),
                borderRadius: BorderRadius.circular(16),
              ),
              child: const Icon(Icons.memory_rounded, color: Color(0xFF43A047), size: 30),
            ),
            const SizedBox(width: 14),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Flexible(
                        child: Text(
                          'Model Status',
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF2D3748),
                          ),
                          overflow: TextOverflow.ellipsis,
                        ),
                      ),
                      const SizedBox(width: 8),
                      Container(
                        padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 3),
                        decoration: BoxDecoration(
                          color: const Color(0xFFE8F5E9),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: const [
                            Icon(Icons.check_circle, color: Color(0xFF43A047), size: 12),
                            SizedBox(width: 4),
                            Text(
                              'Active',
                              style: TextStyle(
                                color: Color(0xFF2E7D32),
                                fontSize: 10,
                                fontWeight: FontWeight.bold,
                              ),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 6),
                  Text(
                    'Model Version: v1.2.3\nLast Updated: 20 May 2025, 08:30 AM',
                    style: TextStyle(
                      fontSize: 11,
                      color: Colors.grey[600],
                      height: 1.3,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(width: 8),
            Icon(Icons.chevron_right, color: Colors.grey.shade400, size: 22),
          ],
        ),
      ),
    );
  }

  Widget _buildBottomNavigationBar() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildNavItem(0, Icons.home_rounded, 'Home', const Color(0xFFE53935)),
          _buildNavItem(1, Icons.qr_code_scanner_rounded, 'Scan', Colors.grey.shade600),
          _buildNavItem(2, Icons.bar_chart_rounded, 'Reports', Colors.grey.shade600),
          _buildNavItem(3, Icons.access_time_rounded, 'History', Colors.grey.shade600),
          _buildNavItem(4, Icons.settings_rounded, 'Settings', Colors.grey.shade600),
        ],
      ),
    );
  }

  Widget _buildNavItem(int index, IconData icon, String label, Color color) {
    bool isSelected = _currentIndex == index;
    return InkWell(
      onTap: () => _onBottomNavTapped(index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: isSelected ? const Color(0xFFE53935) : color, size: 24),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              color: isSelected ? const Color(0xFFE53935) : color,
              fontSize: 10,
              fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }

  void _showInfoDialog(BuildContext context, String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK', style: TextStyle(color: Color(0xFFE53935))),
          ),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// SCAN SCREEN (scan.dart)
// -----------------------------------------------------------------------------
class ScanScreen extends StatelessWidget {
  const ScanScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan Meat'),
        backgroundColor: const Color(0xFFE53935),
        foregroundColor: Colors.white,
        systemOverlayStyle: SystemUiOverlayStyle.light,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pushReplacementNamed(context, '/home'),
        ),
      ),
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(24.0),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: const Color(0xFFFFECEE),
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.qr_code_scanner_rounded, size: 64, color: Color(0xFFE53935)),
              ),
              const SizedBox(height: 24),
              const Text(
                'TFLite Camera Scanner Active',
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Color(0xFF2D3748)),
              ),
              const SizedBox(height: 8),
              Text(
                'Position the beef sample inside the viewfinder to evaluate freshness.',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, color: Colors.grey[600]),
              ),
              const SizedBox(height: 32),
              ElevatedButton.icon(
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFFE53935),
                  padding: const EdgeInsets.symmetric(horizontal: 28, vertical: 14),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                ),
                onPressed: () {
                  ScaffoldMessenger.of(context).showSnackBar(
                    const SnackBar(content: Text('Simulating meat capture & TFLite inference... Result: Fresh (94.2%)')),
                  );
                },
                icon: const Icon(Icons.camera, color: Colors.white),
                label: const Text('Capture & Analyze', style: TextStyle(color: Colors.white, fontSize: 16)),
              ),
            ],
          ),
        ),
      ),
      bottomNavigationBar: _buildStubBottomNav(context, 1),
    );
  }

  Widget _buildStubBottomNav(BuildContext context, int currentIndex) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.05), blurRadius: 10)],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _navItem(context, 0, Icons.home_rounded, 'Home', '/home', currentIndex),
          _navItem(context, 1, Icons.qr_code_scanner_rounded, 'Scan', '/scan', currentIndex),
          _navItem(context, 2, Icons.bar_chart_rounded, 'Reports', '/reports', currentIndex),
          _navItem(context, 3, Icons.access_time_rounded, 'History', '/history', currentIndex),
          _navItem(context, 4, Icons.settings_rounded, 'Settings', '/settings', currentIndex),
        ],
      ),
    );
  }

  Widget _navItem(BuildContext context, int index, IconData icon, String label, String route, int current) {
    bool isSel = current == index;
    return InkWell(
      onTap: () => Navigator.pushReplacementNamed(context, route),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, color: isSel ? const Color(0xFFE53935) : Colors.grey.shade600, size: 24),
          const SizedBox(height: 2),
          Text(label, style: TextStyle(color: isSel ? const Color(0xFFE53935) : Colors.grey.shade600, fontSize: 10, fontWeight: isSel ? FontWeight.bold : FontWeight.normal)),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// REPORTS SCREEN (reports.dart)
// -----------------------------------------------------------------------------
class ReportsScreen extends StatelessWidget {
  const ReportsScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Reports & Analytics'),
        backgroundColor: const Color(0xFFE53935),
        foregroundColor: Colors.white,
        systemOverlayStyle: SystemUiOverlayStyle.light,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pushReplacementNamed(context, '/home'),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text('Meat Inspection Analytics', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Color(0xFF2D3748))),
            const SizedBox(height: 12),
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(16), boxShadow: [BoxShadow(color: Colors.black.withOpacity(0.03), blurRadius: 8)]),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _reportMetric('Total Scans', '142', Colors.blue),
                      _reportMetric('Fresh Rate', '88.7%', Colors.green),
                      _reportMetric('Spoiled', '11.3%', Colors.red),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: 24),
            const Text('Recent Market Inspection Summary', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF2D3748))),
            const SizedBox(height: 12),
            _reportTile('Kumasi Central Market', '95% Fresh • 35 Samples', 'Today, 10:30 AM'),
            _reportTile('Kejetia Meat Shop Block B', '78% Fresh • 50 Samples', 'Yesterday, 02:15 PM'),
            _reportTile('KNUST Abattoir Cold Room', '98% Fresh • 57 Samples', '18 May 2025'),
          ],
        ),
      ),
      bottomNavigationBar: ScanScreen()._buildStubBottomNav(context, 2),
    );
  }

  Widget _reportMetric(String title, String val, Color color) {
    return Column(
      children: [
        Text(val, style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: color)),
        const SizedBox(height: 4),
        Text(title, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
      ],
    );
  }

  Widget _reportTile(String subtitle, String details, String time) {
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(color: Colors.white, borderRadius: BorderRadius.circular(12), border: Border.all(color: Colors.grey.shade200)),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(subtitle, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 14)),
              const SizedBox(height: 4),
              Text(details, style: TextStyle(color: Colors.grey[600], fontSize: 12)),
            ],
          ),
          Text(time, style: TextStyle(color: Colors.grey[400], fontSize: 11)),
        ],
      ),
    );
  }
}

// -----------------------------------------------------------------------------
// HISTORY SCREEN (history.dart)
// -----------------------------------------------------------------------------
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    final List<Map<String, dynamic>> history = [
      {'title': 'Beef Tenderloin Sample #104', 'result': 'FRESH (96.4%)', 'isFresh': true, 'time': 'Today, 11:45 AM'},
      {'title': 'Ribeye Cut Block A', 'result': 'FRESH (91.2%)', 'isFresh': true, 'time': 'Today, 09:20 AM'},
      {'title': 'Ground Chuck Meat Sample', 'result': 'SPOILED (89.5%)', 'isFresh': false, 'time': 'Yesterday, 04:10 PM'},
      {'title': 'Brisket Cut Store #3', 'result': 'FRESH (85.0%)', 'isFresh': true, 'time': '18 May 2025'},
      {'title': 'Sirloin Steak Lot B', 'result': 'SPOILED (92.1%)', 'isFresh': false, 'time': '17 May 2025'},
    ];

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan History'),
        backgroundColor: const Color(0xFFE53935),
        foregroundColor: Colors.white,
        systemOverlayStyle: SystemUiOverlayStyle.light,
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pushReplacementNamed(context, '/home'),
        ),
      ),
      body: ListView.builder(
        padding: const EdgeInsets.all(16),
        itemCount: history.length,
        itemBuilder: (context, index) {
          final item = history[index];
          return Container(
            margin: const EdgeInsets.only(bottom: 12),
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(
              color: Colors.white,
              borderRadius: BorderRadius.circular(16),
              border: Border.all(color: Colors.grey.shade200),
            ),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: item['isFresh'] ? const Color(0xFFE8F5E9) : const Color(0xFFFFECEE),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    item['isFresh'] ? Icons.check_circle_rounded : Icons.warning_rounded,
                    color: item['isFresh'] ? const Color(0xFF43A047) : const Color(0xFFE53935),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(item['title'], style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 15, color: Color(0xFF2D3748))),
                      const SizedBox(height: 4),
                      Text(item['result'], style: TextStyle(fontWeight: FontWeight.w600, fontSize: 12, color: item['isFresh'] ? const Color(0xFF2E7D32) : const Color(0xFFC62828))),
                      const SizedBox(height: 2),
                      Text(item['time'], style: TextStyle(fontSize: 11, color: Colors.grey[500])),
                    ],
                  ),
                ),
                const Icon(Icons.chevron_right, color: Colors.grey),
              ],
            ),
          );
        },
      ),
      bottomNavigationBar: ScanScreen()._buildStubBottomNav(context, 3),
    );
  }
}

// -----------------------------------------------------------------------------
// SETTINGS SCREEN (settings.dart)
// -----------------------------------------------------------------------------
class SettingsScreen extends StatefulWidget {
  const SettingsScreen({super.key});

  @override
  State<SettingsScreen> createState() => _SettingsScreenState();
}

class _SettingsScreenState extends State<SettingsScreen> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();
  bool autoFocus = true;
  bool scanConfirmation = true;
  bool scanAlerts = true;
  String flashMode = 'Auto';
  String scanResolution = 'High (1080p)';
  String historyReminders = 'Weekly';

  void _onBottomNavTapped(int index) {
    switch (index) {
      case 0:
        Navigator.pushReplacementNamed(context, '/home');
        break;
      case 1:
        Navigator.pushReplacementNamed(context, '/scan');
        break;
      case 2:
        Navigator.pushReplacementNamed(context, '/reports');
        break;
      case 3:
        Navigator.pushReplacementNamed(context, '/history');
        break;
      case 4:
        // Already on Settings
        break;
    }
  }

  @override
  Widget build(BuildContext context) {
    return AnnotatedRegion<SystemUiOverlayStyle>(
      value: const SystemUiOverlayStyle(
        statusBarColor: Colors.transparent,
        statusBarIconBrightness: Brightness.dark,
        statusBarBrightness: Brightness.light,
      ),
      child: Scaffold(
        key: _scaffoldKey,
        backgroundColor: const Color(0xFFF9FAFB),
        drawer: _buildToggleMenuDrawer(context),
        body: SafeArea(
          child: Column(
            children: [
              _buildAppBar(context),
              Expanded(
                child: SingleChildScrollView(
                  physics: const BouncingScrollPhysics(),
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 4, vertical: 12),
                        child: Text(
                          'Scan Preferences',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF4A1A1A),
                          ),
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: Colors.grey.shade100),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.02),
                              blurRadius: 10,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Column(
                          children: [
                            _buildSwitchTile(
                              icon: Icons.filter_center_focus_rounded,
                              title: 'Auto Focus',
                              subtitle: 'Automatically focus on meat',
                              value: autoFocus,
                              iconColor: const Color(0xFFE53935),
                              bgColor: const Color(0xFFFFEBEE),
                              onChanged: (val) => setState(() => autoFocus = val),
                            ),
                            const Divider(height: 1, indent: 70),
                            _buildDropdownTile(
                              icon: Icons.flash_on_rounded,
                              title: 'Flash',
                              subtitle: 'Control camera flash',
                              value: flashMode,
                              items: const ['Auto', 'On', 'Off'],
                              iconColor: const Color(0xFFE53935),
                              bgColor: const Color(0xFFFFEBEE),
                              onChanged: (val) => setState(() => flashMode = val!),
                            ),
                            const Divider(height: 1, indent: 70),
                            _buildDropdownTile(
                              icon: Icons.grid_view_rounded,
                              title: 'Scan Resolution',
                              subtitle: 'Set image quality for analysis',
                              value: scanResolution,
                              items: const ['High (1080p)', 'Medium (720p)', 'Standard (480p)'],
                              iconColor: const Color(0xFFE53935),
                              bgColor: const Color(0xFFFFEBEE),
                              onChanged: (val) => setState(() => scanResolution = val!),
                            ),
                            const Divider(height: 1, indent: 70),
                            _buildSwitchTile(
                              icon: Icons.timer_rounded,
                              title: 'Scan Confirmation',
                              subtitle: 'Show preview before analysis',
                              value: scanConfirmation,
                              iconColor: const Color(0xFFE53935),
                              bgColor: const Color(0xFFFFEBEE),
                              onChanged: (val) => setState(() => scanConfirmation = val),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 24),
                      const Padding(
                        padding: EdgeInsets.symmetric(horizontal: 4, vertical: 12),
                        child: Text(
                          'Notifications',
                          style: TextStyle(
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                            color: Color(0xFF4A1A1A),
                          ),
                        ),
                      ),
                      Container(
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(24),
                          border: Border.all(color: Colors.grey.shade100),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.02),
                              blurRadius: 10,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Column(
                          children: [
                            _buildSwitchTile(
                              icon: Icons.notifications_none_rounded,
                              title: 'Scan Alerts',
                              subtitle: 'Get notified about scan results',
                              value: scanAlerts,
                              iconColor: const Color(0xFF1E88E5),
                              bgColor: const Color(0xFFE3F2FD),
                              onChanged: (val) => setState(() => scanAlerts = val),
                            ),
                            const Divider(height: 1, indent: 70),
                            _buildDropdownTile(
                              icon: Icons.history_rounded,
                              title: 'History Reminders',
                              subtitle: 'Remind me to check scan history',
                              value: historyReminders,
                              items: const ['Daily', 'Weekly', 'Monthly', 'Off'],
                              iconColor: const Color(0xFF1E88E5),
                              bgColor: const Color(0xFFE3F2FD),
                              onChanged: (val) => setState(() => historyReminders = val!),
                            ),
                          ],
                        ),
                      ),
                      const SizedBox(height: 30),
                    ],
                  ),
                ),
              ),
              _buildBottomNavigationBar(),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildBottomNavigationBar() {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 8),
      decoration: BoxDecoration(
        color: Colors.white,
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.05),
            blurRadius: 10,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceAround,
        children: [
          _buildNavItem(0, Icons.home_rounded, 'Home', Colors.grey.shade600),
          _buildNavItem(1, Icons.qr_code_scanner_rounded, 'Scan', Colors.grey.shade600),
          _buildNavItem(2, Icons.bar_chart_rounded, 'Reports', Colors.grey.shade600),
          _buildNavItem(3, Icons.access_time_rounded, 'History', Colors.grey.shade600),
          _buildNavItem(4, Icons.settings_rounded, 'Settings', const Color(0xFFE53935)),
        ],
      ),
    );
  }

  Widget _buildNavItem(int index, IconData icon, String label, Color color) {
    return InkWell(
      onTap: () => _onBottomNavTapped(index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: index == 4 ? const Color(0xFFE53935) : color,
            size: 24,
          ),
          const SizedBox(height: 2),
          Text(
            label,
            style: TextStyle(
              color: index == 4 ? const Color(0xFFE53935) : color,
              fontSize: 10,
              fontWeight: index == 4 ? FontWeight.bold : FontWeight.normal,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAppBar(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8.0, vertical: 12.0),
      decoration: const BoxDecoration(
        color: Colors.transparent,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              IconButton(
                icon: const Icon(Icons.menu_rounded, color: Colors.black87, size: 28),
                onPressed: () {
                  _scaffoldKey.currentState?.openDrawer();
                },
              ),
              Expanded(
                child: Center(
                  child: Image.asset(
                    'lib/assets/images/logo.png',
                    height: 48,
                    fit: BoxFit.contain,
                    errorBuilder: (context, error, stackTrace) => const Icon(Icons.shield_rounded, color: Color(0xFFE53935), size: 32),
                  ),
                ),
              ),
              Stack(
                children: [
                  IconButton(
                    icon: const Icon(Icons.notifications_none_rounded, color: Colors.black87, size: 28),
                    onPressed: () {},
                  ),
                  Positioned(
                    right: 8,
                    top: 8,
                    child: Container(
                      padding: const EdgeInsets.all(4),
                      decoration: const BoxDecoration(color: Color(0xFFE53935), shape: BoxShape.circle),
                      child: const Text('2', style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                    ),
                  ),
                ],
              ),
            ],
          ),
          const SizedBox(height: 20),
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Settings',
                  style: TextStyle(
                    fontSize: 32,
                    fontWeight: FontWeight.bold,
                    color: Color(0xFF2D3748),
                    letterSpacing: -1,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Customize your app experience',
                  style: TextStyle(fontSize: 15, color: Colors.grey[600], fontWeight: FontWeight.w400),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildToggleMenuDrawer(BuildContext context) {
    return Drawer(
      child: Column(
        children: [
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                DrawerHeader(
                  decoration: const BoxDecoration(
                    gradient: LinearGradient(
                      colors: [Color(0xFFE53935), Color(0xFFB71C1C)],
                      begin: Alignment.topLeft,
                      end: Alignment.bottomRight,
                    ),
                  ),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(8),
                        decoration: BoxDecoration(
                          color: Colors.white,
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Image.asset(
                          'lib/assets/images/logo.png',
                          height: 48,
                          errorBuilder: (context, error, stackTrace) {
                            return const Icon(Icons.shield_rounded, color: Color(0xFFE53935), size: 40);
                          },
                        ),
                      ),
                    ],
                  ),
                ),
                ListTile(
                  leading: const Icon(Icons.history_rounded, color: Color(0xFFE53935)),
                  title: const Text('Scan history'),
                  onTap: () {
                    Navigator.pop(context);
                    Navigator.pushReplacementNamed(context, '/history');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.memory_rounded, color: Color(0xFFE53935)),
                  title: const Text('Model Center (v1.2.3)'),
                  subtitle: const Text('TFLite Model Active'),
                  onTap: () {
                    Navigator.pop(context);
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Model Center: MeatScan v1.2.3 loaded successfully.')),
                    );
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.settings_rounded, color: Color(0xFFE53935)),
                  title: const Text('Settings'),
                  onTap: () {
                    Navigator.pop(context);
                    // Already on settings, do nothing or just close drawer
                  },
                ),
                const Divider(),
                ListTile(
                  leading: const Icon(Icons.help_outline_rounded, color: Colors.grey),
                  title: const Text('Support & Help Center'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'Support & Help Center', 'Contact support at support@meatguard.ai or visit our documentation portal.');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.info_outline_rounded, color: Colors.grey),
                  title: const Text('About MeatGuard'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'About MeatGuard', 'MeatGuard is an advanced AI-driven beef freshness detection app built for reliable field inspection using TFLite.');
                  },
                ),
                ListTile(
                  leading: const Icon(Icons.bug_report_outlined, color: Colors.grey),
                  title: const Text('Feedback & Bug Report'),
                  onTap: () {
                    Navigator.pop(context);
                    _showInfoDialog(context, 'Feedback & Bug Report', 'Thank you for helping us improve! Send bug reports to feedback@meatguard.ai');
                  },
                ),
              ],
            ),
          ),
          Padding(
            padding: const EdgeInsets.all(20.0),
            child: Column(
              children: [
                const Text(
                  'MeatGuard AI',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                const Text(
                  'Version 1.2.3 (Production)',
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 11,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  void _showInfoDialog(BuildContext context, String title, String message) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(title),
        content: Text(message),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK', style: TextStyle(color: Color(0xFFE53935))),
          ),
        ],
      ),
    );
  }

  Widget _buildSwitchTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required bool value,
    required Color iconColor,
    required Color bgColor,
    required ValueChanged<bool> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: iconColor, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF2D3748)),
                ),
                const SizedBox(height: 2),
                Text(subtitle, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
              ],
            ),
          ),
          Switch.adaptive(
            value: value,
            activeTrackColor: const Color(0xFFE53935).withValues(alpha: 0.5),
            activeThumbColor: const Color(0xFFE53935),
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }

  Widget _buildDropdownTile({
    required IconData icon,
    required String title,
    required String subtitle,
    required String value,
    required List<String> items,
    required Color iconColor,
    required Color bgColor,
    required ValueChanged<String?> onChanged,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: bgColor,
              borderRadius: BorderRadius.circular(16),
            ),
            child: Icon(icon, color: iconColor, size: 24),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.bold, color: Color(0xFF2D3748)),
                ),
                const SizedBox(height: 2),
                Text(subtitle, style: TextStyle(fontSize: 12, color: Colors.grey[600])),
              ],
            ),
          ),
          DropdownButton<String>(
            value: value,
            underline: const SizedBox(),
            icon: const Icon(Icons.expand_more_rounded, color: Color(0xFF632020)),
            style: const TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Color(0xFF632020)),
            items: items.map((String item) {
              return DropdownMenuItem<String>(
                value: item,
                child: Text(item),
              );
            }).toList(),
            onChanged: onChanged,
          ),
        ],
      ),
    );
  }
}