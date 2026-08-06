import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import 'scan.dart';
import 'history.dart';
import 'reports.dart';
import 'settings.dart';

void main() {
  runApp(const MeatGuardApp());
}

class MeatGuardApp extends StatelessWidget {
  const MeatGuardApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'MeatGuard',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(
          seedColor: const Color(0xFFC62828), // Deep red theme color
          background: const Color(0xFFF8F9FA),
        ),
        useMaterial3: true,
        fontFamily: 'Roboto', // Assuming a standard font, adjust if needed
      ),
      home: const HomeScreen(),
      debugShowCheckedModeBanner: false,
    );
  }
}

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  // Key for the scaffold to open the drawer
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: Theme.of(context).colorScheme.background,
      appBar: AppBar(
        systemOverlayStyle: SystemUiOverlayStyle.dark, // Makes status bar icons visible on white background
        backgroundColor: Colors.transparent,
        elevation: 0,
        leading: IconButton(
          icon: const Icon(Icons.menu, color: Colors.black87, size: 28),
          onPressed: () {
            _scaffoldKey.currentState?.openDrawer();
          },
        ),
        title: Row(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            // Logo image containing both icon and text
            Image.asset(
              'lib/assets/images/logo.png',
              height: 40, // Adjust height as needed based on your logo asset
              fit: BoxFit.contain,
              errorBuilder: (context, error, stackTrace) {
                // Fallback if logo is missing during development
                return const Text('MeatGuard', style: TextStyle(color: Colors.red, fontWeight: FontWeight.bold));
              },
            ),
          ],
        ),
        actions: [
          // Notification icon with badge
          Stack(
            alignment: Alignment.center,
            children: [
              IconButton(
                icon: const Icon(Icons.notifications_none, color: Colors.black87, size: 28),
                onPressed: () {},
              ),
              Positioned(
                top: 12,
                right: 12,
                child: Container(
                  padding: const EdgeInsets.all(4),
                  decoration: const BoxDecoration(
                    color: Colors.red,
                    shape: BoxShape.circle,
                  ),
                  child: const Text(
                    '2',
                    style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold),
                  ),
                ),
              )
            ],
          ),
          const SizedBox(width: 8),
        ],
      ),
      drawer: Drawer(
        child: ListView(
          padding: EdgeInsets.zero,
          children: [
            DrawerHeader(
              decoration: const BoxDecoration(
                color: Color(0xFFFBE9E7), // Light red background
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                mainAxisAlignment: MainAxisAlignment.end,
                children: [
                  Image.asset(
                    'lib/assets/images/logo.png',
                    height: 50,
                    errorBuilder: (context, error, stackTrace) => const Icon(Icons.shield, color: Colors.red, size: 50),
                  ),
                  const SizedBox(height: 10),
                  const Text(
                    'AI Powered Beef Freshness Detection',
                    style: TextStyle(color: Colors.black54, fontSize: 14),
                  ),
                ],
              ),
            ),
            ListTile(
              leading: const Icon(Icons.history),
              title: const Text('Scan history'),
              onTap: () {
                Navigator.pop(context); // Close drawer
                Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryScreen()));
              },
            ),
            ListTile(
              leading: const Icon(Icons.memory),
              title: const Text('Model Center (v1.2.3)'),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: const Icon(Icons.settings),
              title: const Text('Settings'),
              onTap: () {
                Navigator.pop(context); // Close drawer
                Navigator.push(context, MaterialPageRoute(builder: (context) => const SettingsScreen()));
              },
            ),
            const Divider(),
            ListTile(
              leading: const Icon(Icons.help_outline),
              title: const Text('Support & Help Center'),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: const Icon(Icons.info_outline),
              title: const Text('About MeatGuard'),
              onTap: () {
                Navigator.pop(context);
              },
            ),
            ListTile(
              leading: const Icon(Icons.bug_report_outlined),
              title: const Text('Feedback & Bug Report'),
              onTap: () {
                Navigator.pop(context);
              },
            ),
          ],
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // 1. Promotional/Action Banner
            _buildActionBanner(),
            const SizedBox(height: 24),

            // 2. Grid for main actions (Scan, History, Reports, Settings)
            GridView.count(
              crossAxisCount: 2,
              crossAxisSpacing: 16,
              mainAxisSpacing: 16,
              shrinkWrap: true,
              physics: const NeverScrollableScrollPhysics(),
              children: [
                _buildActionCard(
                  icon: Icons.document_scanner_outlined,
                  iconColor: Colors.red.shade200,
                  iconBgColor: Colors.red.shade50,
                  title: 'Scan Meat',
                  subtitle: 'Scan a meat sample to check if it is fresh or spoiled',
                  onTap: () {
                    // Navigate to Scan screen
                  },
                ),
                _buildActionCard(
                  icon: Icons.history,
                  iconColor: Colors.blue.shade200,
                  iconBgColor: Colors.blue.shade50,
                  title: 'Scan History',
                  subtitle: 'View your previous scans and results',
                  onTap: () {
                    // Navigate to History screen
                  },
                ),
                _buildActionCard(
                  icon: Icons.bar_chart,
                  iconColor: Colors.green.shade200,
                  iconBgColor: Colors.green.shade50,
                  title: 'Reports',
                  subtitle: 'See analytics and insights about your scans',
                  onTap: () {},
                ),
                _buildActionCard(
                  icon: Icons.settings_outlined,
                  iconColor: Colors.purple.shade200,
                  iconBgColor: Colors.purple.shade50,
                  title: 'Settings',
                  subtitle: 'Configure app preferences and options',
                  onTap: () {
                    // Navigate to Settings screen
                  },
                ),
              ],
            ),
            const SizedBox(height: 24),

            // 3. Model Status Card at the bottom
            _buildModelStatusCard(),
            const SizedBox(height: 20), // Extra padding at bottom
          ],
        ),
      ),
      bottomNavigationBar: BottomNavigationBar(
        type: BottomNavigationBarType.fixed,
        selectedItemColor: Colors.red.shade700,
        unselectedItemColor: Colors.grey,
        currentIndex: 0, // Set to Home
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.home), label: 'Home'),
          BottomNavigationBarItem(icon: Icon(Icons.document_scanner_outlined), label: 'Scan'),
          BottomNavigationBarItem(icon: Icon(Icons.bar_chart), label: 'Reports'),
          BottomNavigationBarItem(icon: Icon(Icons.history), label: 'History'),
          BottomNavigationBarItem(icon: Icon(Icons.settings_outlined), label: 'Settings'),
        ],
        onTap: (index) {
          // Handle navigation here based on index
          if (index == 1) {
            Navigator.push(context, MaterialPageRoute(builder: (context) => const ScanScreen()));
          } else if (index == 2) {
            Navigator.push(context, MaterialPageRoute(builder: (context) => const ReportsScreen()));
          } else if (index == 3) {
            Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryScreen()));
          } else if (index == 4) {
            Navigator.push(context, MaterialPageRoute(builder: (context) => const SettingsScreen()));
          }
        },
      ),
    );
  }

  Widget _buildActionBanner() {
    return Container(
      height: 230, // Increased height to fix bottom overflow by 16 pixels
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(20),
        image: const DecorationImage(
          image: AssetImage('lib/assets/images/banner.png'),
          fit: BoxFit.cover,
        ),
        // Adding a slight gradient to ensure text is readable over the image
        gradient: LinearGradient(
          begin: Alignment.centerLeft,
          end: Alignment.centerRight,
          colors: [
            Colors.black.withOpacity(0.7),
            Colors.black.withOpacity(0.3),
          ],
        ),
      ),
      child: Stack(
        children: [
          // Fallback gradient if image fails to load
          Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(20),
              gradient: LinearGradient(
                begin: Alignment.centerLeft,
                end: Alignment.centerRight,
                colors: [
                  Colors.black.withOpacity(0.8),
                  Colors.transparent,
                ],
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
                  padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                  decoration: BoxDecoration(
                    color: Colors.red.withOpacity(0.8),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      Icon(Icons.auto_awesome, color: Colors.white, size: 12),
                      SizedBox(width: 4),
                      Text('AI POWERED', style: TextStyle(color: Colors.white, fontSize: 10, fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                const Text(
                  'Detect Freshness.\nProtect Health.',
                  style: TextStyle(
                    color: Colors.white,
                    fontSize: 22,
                    fontWeight: FontWeight.bold,
                    height: 1.2,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'Instantly scan meat and get\nreliable AI-powered results.',
                  style: TextStyle(
                    color: Colors.white70,
                    fontSize: 12,
                  ),
                ),
                const Spacer(),
                ElevatedButton.icon(
                  onPressed: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const ScanScreen()));
                  },
                  icon: const Icon(Icons.document_scanner_outlined, color: Colors.black),
                  label: const Text('SCAN MEAT NOW', style: TextStyle(color: Colors.black, fontWeight: FontWeight.bold)),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.white,
                    foregroundColor: Colors.black,
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                  ),
                ),
              ],
            ),
          ),
          // Pagination dots (static for visual match)
          Positioned(
            bottom: 20,
            right: 20,
            child: Row(
              children: [
                _buildActionCard(
                  icon: Icons.document_scanner_outlined,
                  iconColor: Colors.red.shade200,
                  iconBgColor: Colors.red.shade50,
                  title: 'Scan Meat',
                  subtitle: 'Scan a meat sample to check if it is fresh or spoiled',
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const ScanScreen()));
                  },
                ),
                _buildActionCard(
                  icon: Icons.history,
                  iconColor: Colors.blue.shade200,
                  iconBgColor: Colors.blue.shade50,
                  title: 'Scan History',
                  subtitle: 'View your previous scans and results',
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const HistoryScreen()));
                  },
                ),
                _buildActionCard(
                  icon: Icons.bar_chart,
                  iconColor: Colors.green.shade200,
                  iconBgColor: Colors.green.shade50,
                  title: 'Reports',
                  subtitle: 'See analytics and insights about your scans',
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const ReportsScreen()));
                  },
                ),
                _buildActionCard(
                  icon: Icons.settings_outlined,
                  iconColor: Colors.purple.shade200,
                  iconBgColor: Colors.purple.shade50,
                  title: 'Settings',
                  subtitle: 'Configure app preferences and options',
                  onTap: () {
                    Navigator.push(context, MaterialPageRoute(builder: (context) => const SettingsScreen()));
                  },
                ),
              ],
            ),
          )
        ],
      ),
    );
  }

  Widget _buildActionCard({
    required IconData icon,
    required Color iconColor,
    required Color iconBgColor,
    required String title,
    required String subtitle,
    required VoidCallback onTap,
  }) {
    return GestureDetector(
      onTap: onTap,
      child: Container(
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(16),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.1),
              spreadRadius: 1,
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: iconBgColor,
                    shape: BoxShape.circle,
                  ),
                  child: Icon(icon, color: iconColor, size: 24),
                ),
                Icon(Icons.arrow_forward_ios, color: Colors.grey.shade400, size: 14),
              ],
            ),
            const Spacer(),
            Text(
              title,
              style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 16),
            ),
            const SizedBox(height: 4),
            Text(
              subtitle,
              style: TextStyle(color: Colors.grey.shade600, fontSize: 11, height: 1.2),
              maxLines: 2,
              overflow: TextOverflow.ellipsis,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildModelStatusCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.grey.withOpacity(0.1),
            spreadRadius: 1,
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(12),
            decoration: BoxDecoration(
              color: Colors.green.shade50,
              borderRadius: BorderRadius.circular(12),
            ),
            child: Icon(Icons.memory, color: Colors.green.shade400, size: 30),
          ),
          const SizedBox(width: 16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Model Status', style: TextStyle(fontWeight: FontWeight.bold, fontSize: 16)),
                const SizedBox(height: 4),
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                      decoration: BoxDecoration(
                        color: Colors.green.shade50,
                        borderRadius: BorderRadius.circular(4),
                      ),
                      child: const Row(
                        children: [
                          Icon(Icons.check_circle, color: Colors.green, size: 12),
                          SizedBox(width: 4),
                          Text('Model is Active', style: TextStyle(color: Colors.green, fontSize: 10, fontWeight: FontWeight.bold)),
                        ],
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 4),
                Text('Model Version: v1.2.3', style: TextStyle(color: Colors.grey.shade500, fontSize: 11)),
                Text('Last Updated: 20 May 2025, 08:30 AM', style: TextStyle(color: Colors.grey.shade500, fontSize: 11)),
              ],
            ),
          ),
          Icon(Icons.arrow_forward_ios, color: Colors.grey.shade400, size: 16),
        ],
      ),
    );
  }
}