import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

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
  int _currentIndex = 4; // Settings tab index

  void _onBottomNavTapped(int index) {
    setState(() {
      _currentIndex = index;
    });

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
                              color: Colors.black.withValues(alpha: 0.02),
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
                              color: Colors.black.withValues(alpha: 0.02),
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
    bool isSelected = _currentIndex == index;
    return InkWell(
      onTap: () => _onBottomNavTapped(index),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            color: isSelected ? const Color(0xFFE53935) : color,
            size: 24,
          ),
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
}